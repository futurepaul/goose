//! `GOOSE_NO_TURN_CONTEXT`. A test binary of its own, because the switch is a
//! process environment variable and would leak into the tests running beside
//! it.

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{Agent, SessionConfig};
use goose::config::GooseMode;
use goose::conversation::message::Message;
use goose::conversation::TURN_CONTEXT_TAG;
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::session_manager::SessionType;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

/// Answers every request with words, and keeps what each one sent (the turn
/// context, when there is one, merged into the user's message).
/// What one request sent: its system prompt and its messages.
type Request = (String, Vec<Message>);

#[derive(Default)]
struct RecordingProvider {
    requests: Arc<Mutex<Vec<Request>>>,
}

#[async_trait]
impl Provider for RecordingProvider {
    async fn stream(
        &self,
        _model_config: &ModelConfig,
        system_prompt: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.requests
            .lock()
            .unwrap()
            .push((system_prompt.to_string(), messages.to_vec()));
        Ok(stream_from_single_message(
            Message::assistant().with_text("done"),
            ProviderUsage::new(
                "mock-model".to_string(),
                Usage::new(Some(100), Some(10), Some(110)),
            ),
        ))
    }

    fn get_name(&self) -> &str {
        "mock-recording"
    }
}

/// One turn with `GOOSE_NO_TURN_CONTEXT` set to `switch` (unset when `None`),
/// goose's state under `root` (one for the whole test: goose keeps its
/// session database open once it has opened it): what each of its requests
/// sent (the system prompt, the messages).
async fn one_turn(root: &Path, switch: Option<&str>) -> Result<Vec<Request>> {
    let temp_dir = TempDir::new()?;
    let _guard = env_lock::lock_env([
        ("GOOSE_NO_TURN_CONTEXT", switch),
        ("GOOSE_PATH_ROOT", Some(root.to_str().unwrap())),
        ("GOOSE_DISABLE_KEYRING", Some("1")),
        ("GOOSE_DISABLE_SESSION_NAMING", Some("true")),
    ]);
    let agent = Agent::new();
    let session = agent
        .config
        .session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "no-turn-context".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let provider = RecordingProvider::default();
    let requests = provider.requests.clone();
    agent
        .update_provider(
            Arc::new(provider),
            ModelConfig::new("mock-model").with_context_limit(Some(200_000)),
            &session.id,
        )
        .await?;
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
    };
    let reply_stream = agent
        .reply(Message::user().with_text("hello"), session_config, None)
        .await?;
    tokio::pin!(reply_stream);
    while let Some(event) = reply_stream.next().await {
        event?;
    }
    let requests = requests.lock().unwrap().clone();
    Ok(requests)
}

/// Goal: with `GOOSE_NO_TURN_CONTEXT`, a request carries no `<turn-context>`
/// message and its system prompt says nothing of one: what the host sent is
/// all the model sees after goose's own prompt. Control: without the switch
/// (and with it set to something else), the same turn has both.
#[tokio::test]
async fn no_turn_context_when_switched_off() -> Result<()> {
    let root = TempDir::new()?;
    let open_tag = format!("<{TURN_CONTEXT_TAG}>");
    for control in [None, Some("0")] {
        let requests = one_turn(root.path(), control).await?;
        let (system, messages) = requests.first().expect("the turn made a request");
        assert!(system.contains("# Turn Context"), "control ({control:?})");
        assert!(
            messages
                .iter()
                .any(|m| m.as_concat_text().contains(&open_tag)),
            "control ({control:?}): {messages:?}"
        );
    }

    for switch in ["1", "true"] {
        let requests = one_turn(root.path(), Some(switch)).await?;
        assert!(!requests.is_empty(), "{switch}");
        for (system, messages) in &requests {
            assert!(!system.contains("# Turn Context"), "{switch}: {system}");
            assert!(!system.contains(&open_tag), "{switch}");
            assert!(
                messages
                    .iter()
                    .all(|m| !m.as_concat_text().contains(&open_tag)),
                "{switch}: {messages:?}"
            );
            assert_eq!(
                messages
                    .iter()
                    .map(Message::as_concat_text)
                    .collect::<Vec<_>>(),
                vec!["hello".to_string()],
                "the host's message alone ({switch})"
            );
        }
    }

    Ok(())
}
