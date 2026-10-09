//! `GOOSE_NO_COMPACTION`. A test binary of its own, because the switch is a
//! process environment variable and would leak into the compaction tests
//! running beside it.

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{Agent, AgentEvent, SessionConfig};
use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageContent, MessageErrorKind};
use goose::conversation::Conversation;
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::session_manager::SessionType;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tempfile::TempDir;

/// Overflows on every turn request; answers a compaction request (the only
/// kind that asks to summarize) with a summary, and counts each kind.
#[derive(Default)]
struct OverflowingProvider {
    summarize_requests: Arc<AtomicUsize>,
    turn_requests: Arc<AtomicUsize>,
}

#[async_trait]
impl Provider for OverflowingProvider {
    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system_prompt: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let summarize = messages.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(content, MessageContent::Text(text)
                    if text.text.to_lowercase().contains("summarize"))
            })
        });
        if !summarize {
            self.turn_requests.fetch_add(1, Ordering::SeqCst);
            return Err(ProviderError::ContextLengthExceeded(
                "prompt is too long".to_string(),
            ));
        }
        self.summarize_requests.fetch_add(1, Ordering::SeqCst);
        Ok(stream_from_single_message(
            Message::assistant().with_text("<mock summary>"),
            ProviderUsage::new(
                "mock-model".to_string(),
                Usage::new(Some(100), Some(10), Some(110)),
            ),
        ))
    }

    fn get_name(&self) -> &str {
        "mock-overflowing"
    }
}

struct Turn {
    history_replaced: bool,
    messages: Vec<Message>,
    summarize_requests: usize,
    turn_requests: usize,
    stored: Vec<Message>,
}

/// One turn of a session far over any auto-compact threshold whose provider
/// overflows, with `GOOSE_NO_COMPACTION` set to `switch` (unset when `None`)
/// and tool-pair summarization asked for; goose's state under `root` (one for
/// the whole test: goose keeps its session database open once it has opened
/// it).
async fn overflowing_turn(root: &Path, switch: Option<&str>, history: &[Message]) -> Result<Turn> {
    let temp_dir = TempDir::new()?;
    let _guard = env_lock::lock_env([
        ("GOOSE_NO_COMPACTION", switch),
        ("GOOSE_TOOL_PAIR_SUMMARIZATION", Some("true")),
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
            "no-compaction".to_string(),
            SessionType::Hidden,
            GooseMode::default(),
        )
        .await?;
    agent
        .config
        .session_manager
        .replace_conversation(
            &session.id,
            &Conversation::new_unvalidated(history.to_vec()),
        )
        .await?;
    agent
        .config
        .session_manager
        .update(&session.id)
        .usage(Usage::new(Some(999_000), Some(1_000), Some(1_000_000)))
        .apply()
        .await?;

    let provider = OverflowingProvider::default();
    let (summarize_requests, turn_requests) = (
        provider.summarize_requests.clone(),
        provider.turn_requests.clone(),
    );
    agent
        .update_provider(
            Arc::new(provider),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
    };
    let reply_stream = agent
        .reply(
            Message::user().with_text("Tell me more"),
            session_config,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);
    let (mut history_replaced, mut messages) = (false, Vec::new());
    while let Some(event) = reply_stream.next().await {
        match event? {
            AgentEvent::HistoryReplaced(_) => history_replaced = true,
            AgentEvent::Message(message) => messages.push(message),
            _ => {}
        }
    }

    let stored = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?
        .conversation
        .expect("session has a conversation")
        .messages()
        .to_vec();
    Ok(Turn {
        history_replaced,
        messages,
        summarize_requests: summarize_requests.load(Ordering::SeqCst),
        turn_requests: turn_requests.load(Ordering::SeqCst),
        stored,
    })
}

/// Goal: with `GOOSE_NO_COMPACTION`, a session over the threshold whose
/// provider overflows is never compacted (ahead of the turn, or to recover
/// from the overflow): the turn asks once, ends with the provider's error,
/// and what the host sent stays as it was. Control: without the switch (and
/// with it set to something else), goose compacts this session.
#[tokio::test]
async fn overflow_ends_the_turn_without_compacting() -> Result<()> {
    let root = TempDir::new()?;
    let history = vec![
        Message::user().with_text("Hello"),
        Message::assistant().with_text("Hi there"),
    ];

    for control in [None, Some("0")] {
        let turn = overflowing_turn(root.path(), control, &history).await?;
        assert!(
            turn.summarize_requests > 0,
            "control ({control:?}): without the switch, goose compacts this session"
        );
    }

    for switch in ["1", "true"] {
        let turn = overflowing_turn(root.path(), Some(switch), &history).await?;
        assert!(!turn.history_replaced, "{switch}");
        assert_eq!(
            turn.summarize_requests, 0,
            "no compaction request ({switch})"
        );
        assert_eq!(
            turn.turn_requests, 1,
            "asked once, never retried ({switch})"
        );
        assert!(
            turn.messages.iter().any(
                |message| message.error_kind() == Some(MessageErrorKind::ContextLengthExceeded)
            ),
            "the turn ends saying why ({switch}): {:?}",
            turn.messages
        );
        assert!(
            turn.stored
                .iter()
                .all(|message| !message.as_concat_text().contains("mock summary")),
            "nothing was summarized ({switch})"
        );
        for (stored, sent) in turn.stored.iter().zip(&history) {
            assert_eq!(stored.as_concat_text(), sent.as_concat_text(), "{switch}");
            assert!(
                stored.is_agent_visible(),
                "history stays visible ({switch})"
            );
        }
    }

    Ok(())
}
