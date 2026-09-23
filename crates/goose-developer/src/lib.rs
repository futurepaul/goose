//! goose's developer tools (shell, write, edit, tree) as a GDK crate.
//!
//! `shell.rs`, `shell_output_streaming.rs`, `edit.rs` and `tree.rs` are
//! goose's own files from `goose/src/agents/platform_extensions/developer`,
//! with goose's internals replaced by the small shims below, so an embedder
//! can run goose's tools without the `goose` crate. `Developer` plugs into
//! `goose_agent::tool::ToolOperation` as a `ToolProvider`.

pub mod edit;
pub mod shell;
mod shell_output_streaming;
mod subprocess;
pub mod tree;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use edit::{EditTools, FileEditParams, FileWriteParams};
use goose_agent::operation::Emitter;
use goose_agent::tool::ToolProvider;
use rmcp::model::{
    Annotations, CallToolRequestParams, CallToolResult, ContentBlock, ErrorData, JsonObject,
    ServerNotification, TextContent, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde_json::Value;
use shell::{shell_display_name, ShellOutput, ShellParams, ShellTool};
use tokio_util::sync::CancellationToken;
use tree::{TreeParams, TreeTool};

pub use shell_output_streaming::parse_shell_output_notification;

/// goose's `DEFAULT_EXTENSION_TIMEOUT`; `GOOSE_DEFAULT_EXTENSION_TIMEOUT`
/// overrides it, as in goose.
pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 300;

/// Sends tool progress (live shell output) to whoever drives the loop,
/// never blocking the tool on a slow consumer.
#[derive(Clone)]
pub struct ToolCallNotificationEmitter {
    sender: tokio::sync::mpsc::Sender<ServerNotification>,
}

impl ToolCallNotificationEmitter {
    pub fn new(sender: tokio::sync::mpsc::Sender<ServerNotification>) -> Self {
        Self { sender }
    }

    pub(crate) fn emit_best_effort(&self, notification: ServerNotification) {
        let _ = self.sender.try_send(notification);
    }
}

pub fn instructions() -> &'static str {
    "Use the developer tools to build software and operate a terminal. Make sure to use the \
tools *efficiently* - reading all the content you need in as few iterations as possible and \
then making the requested edits or running commands. You are responsible for managing your \
context window, and to minimize unnecessary turns which cost the user money.\n\nFor editing \
software, prefer the flow of using tree to understand the codebase structure and file sizes. \
When you need to search, prefer rg which correctly respects gitignored content. Then use cat \
or sed to gather the context you need, always reading before editing. Use write and edit to \
efficiently make changes. Test and verify as appropriate. When running Python scripts or \
commands, always use `python3` instead of `python`."
}

fn visible_text(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text(
        TextContent::new(text).with_annotations(Annotations::default().with_priority(0.0)),
    )
}

fn schema<T: JsonSchema>() -> JsonObject {
    serde_json::to_value(schema_for!(T))
        .expect("schema serialization should succeed")
        .as_object()
        .expect("schema should serialize to an object")
        .clone()
}

fn parse_args<T: serde::de::DeserializeOwned>(arguments: Option<JsonObject>) -> Result<T, String> {
    let value = arguments
        .map(Value::Object)
        .ok_or_else(|| "Missing arguments".to_string())?;
    serde_json::from_value(value).map_err(|e| format!("Failed to parse arguments: {e}"))
}

/// goose's developer tools, rooted at one working directory.
pub struct Developer {
    working_dir: Option<PathBuf>,
    session_id: String,
    emitter: Option<ToolCallNotificationEmitter>,
    shell: Arc<ShellTool>,
    edit: EditTools,
    tree: TreeTool,
}

impl Developer {
    pub fn new(working_dir: Option<PathBuf>, session_id: impl Into<String>) -> std::io::Result<Self> {
        Ok(Self {
            working_dir,
            session_id: session_id.into(),
            emitter: None,
            shell: Arc::new(ShellTool::new(false)?),
            edit: EditTools::new(),
            tree: TreeTool::new(),
        })
    }

    pub fn with_emitter(mut self, emitter: ToolCallNotificationEmitter) -> Self {
        self.emitter = Some(emitter);
        self
    }

    pub fn tools() -> Vec<Tool> {
        let shell = shell_display_name();
        let newline_note = if shell == "cmd" {
            " Commands must be on a single line — cmd.exe silently truncates at the first \
             newline. Use `&` to chain (e.g. `echo a & echo b`) or set GOOSE_SHELL=powershell \
             for multi-line support."
        } else {
            ""
        };
        let shell_description = format!(
            "Execute a shell command in the current dir. Commands run under `{shell}` (set \
             GOOSE_SHELL to override) - write command strings in that shell's syntax.\
             {newline_note} Returns an object with stdout and stderr as separate fields. The \
             output of each stream is limited to up to 2000 lines, and longer outputs will be \
             saved to a temporary file.",
        );
        vec![
            Tool::new(
                "write".to_string(),
                "Create a new file or overwrite an existing file. Creates parent directories if needed.".to_string(),
                schema::<FileWriteParams>(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Write".to_string()),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "edit".to_string(),
                "Edit a file by finding and replacing text. The before text must match exactly and uniquely. Use empty after text to delete.".to_string(),
                schema::<FileEditParams>(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Edit".to_string()),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
            )),
            Tool::new("shell".to_string(), shell_description, schema::<ShellParams>())
                .with_output_schema::<ShellOutput>()
                .annotate(ToolAnnotations::from_raw(
                    Some("Shell".to_string()),
                    Some(false),
                    Some(true),
                    Some(false),
                    Some(true),
                )),
            Tool::new(
                "tree".to_string(),
                "List a directory tree with line counts. Traversal respects .gitignore rules.".to_string(),
                schema::<TreeParams>(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Tree".to_string()),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
            )),
        ]
    }

    /// goose's `DeveloperClient::call_tool` dispatch, minus `read_image`.
    pub async fn call(
        &self,
        name: &str,
        arguments: Option<JsonObject>,
        cancel: CancellationToken,
    ) -> CallToolResult {
        let working_dir = self.working_dir.as_deref();
        match name {
            "shell" => match parse_args::<ShellParams>(arguments) {
                Ok(params) => {
                    self.shell
                        .shell_with_cwd_and_emitter(
                            params,
                            working_dir,
                            Some(&self.session_id),
                            self.emitter.clone(),
                            cancel,
                        )
                        .await
                }
                Err(error) => ShellTool::error_result(&format!("Error: {error}"), None),
            },
            "write" => match parse_args::<FileWriteParams>(arguments) {
                Ok(params) => self.edit.file_write_with_cwd(params, working_dir),
                Err(error) => CallToolResult::error(vec![visible_text(format!("Error: {error}"))]),
            },
            "edit" => match parse_args::<FileEditParams>(arguments) {
                Ok(params) => self.edit.file_edit_with_cwd(params, working_dir),
                Err(error) => CallToolResult::error(vec![visible_text(format!("Error: {error}"))]),
            },
            "tree" => match parse_args::<TreeParams>(arguments) {
                Ok(params) => self.tree.tree_with_cwd(params, working_dir),
                Err(error) => CallToolResult::error(vec![visible_text(format!("Error: {error}"))]),
            },
            _ => CallToolResult::error(vec![visible_text(format!("Error: Unknown tool: {name}"))]),
        }
    }
}

#[async_trait]
impl<S: Send + Sync> ToolProvider<S> for Developer {
    async fn tools(&self, _session: &S) -> anyhow::Result<Vec<Tool>> {
        Ok(Self::tools())
    }

    async fn call(
        &self,
        _session: &S,
        _request_id: &str,
        call: CallToolRequestParams,
        emit: &Emitter,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(Developer::call(self, &call.name, call.arguments, emit.cancel_token().clone()).await)
    }
}
