//! fragment/optmem: `GOOSE_NO_COMPACTION`. A test binary of its own, because
//! the switch is a process environment variable and would leak into the
//! compaction tests running beside it.

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{Agent, AgentEvent, SessionConfig};
use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageContent};
use goose::conversation::Conversation;
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::session_manager::SessionType;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tempfile::TempDir;

/// Overflows on every turn request; answers a compaction request (the only
/// kind that asks to summarize) with a summary, and records that it came.
struct OverflowingProvider {
    asked_to_summarize: Arc<AtomicBool>,
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
            return Err(ProviderError::ContextLengthExceeded(
                "prompt is too long".to_string(),
            ));
        }
        self.asked_to_summarize.store(true, Ordering::SeqCst);
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

/// A session over the auto-compact threshold whose provider overflows: goose
/// compacts it (before the turn, and to recover from the overflow) unless
/// GOOSE_NO_COMPACTION is set. With it, in both agent loops, the turn ends and
/// nothing is summarized, replaced or hidden.
#[tokio::test]
async fn overflow_ends_the_turn_without_compacting() -> Result<()> {
    // (no_compaction, use_state_machine); the first is the control.
    for (no_compaction, use_state_machine) in [(false, false), (true, false), (true, true)] {
        let _guard = env_lock::lock_env([("GOOSE_NO_COMPACTION", no_compaction.then_some("1"))]);
        let temp_dir = TempDir::new()?;
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
        let history = vec![
            Message::user().with_text("Hello"),
            Message::assistant().with_text("Hi there"),
        ];
        agent
            .config
            .session_manager
            .replace_conversation(&session.id, &Conversation::new_unvalidated(history.clone()))
            .await?;
        // Far over any auto-compact threshold.
        agent
            .config
            .session_manager
            .update(&session.id)
            .usage(Usage::new(Some(999_000), Some(1_000), Some(1_000_000)))
            .apply()
            .await?;

        let asked_to_summarize = Arc::new(AtomicBool::new(false));
        let provider = OverflowingProvider {
            asked_to_summarize: asked_to_summarize.clone(),
        };
        agent
            .update_provider(
                Arc::new(provider),
                ModelConfig::new("mock-model"),
                &session.id,
            )
            .await?;

        let reply_stream = agent
            .reply(
                Message::user().with_text("Tell me more"),
                SessionConfig {
                    id: session.id.clone(),
                    schedule_id: None,
                    max_turns: None,
                    retry_config: None,
                },
                use_state_machine,
                None,
            )
            .await?;
        tokio::pin!(reply_stream);

        let mut history_replaced = false;
        let mut texts = Vec::new();
        while let Some(event) = reply_stream.next().await {
            match event {
                Ok(AgentEvent::HistoryReplaced(_)) => history_replaced = true,
                Ok(AgentEvent::Message(message)) => texts.push(message.as_concat_text()),
                Ok(_) => {}
                Err(_) => break,
            }
        }

        if !no_compaction {
            assert!(
                asked_to_summarize.load(Ordering::SeqCst),
                "control: without the switch, goose compacts this session"
            );
            continue;
        }

        let label = format!("state machine: {use_state_machine}");
        assert!(!history_replaced, "{label}");
        assert!(
            !asked_to_summarize.load(Ordering::SeqCst),
            "no compaction request reached the provider ({label})"
        );
        if !use_state_machine {
            assert!(
                texts
                    .iter()
                    .any(|text| text.contains("compaction is disabled (GOOSE_NO_COMPACTION)")),
                "the turn says why it ended: {texts:?}"
            );
        }

        let stored = agent
            .config
            .session_manager
            .get_session(&session.id, true)
            .await?
            .conversation
            .expect("session has a conversation");
        assert!(
            stored
                .messages()
                .iter()
                .all(|message| !message.as_concat_text().contains("mock summary")),
            "nothing was summarized ({label})"
        );
        for (stored, sent) in stored.messages().iter().zip(&history) {
            assert_eq!(stored.as_concat_text(), sent.as_concat_text(), "{label}");
            assert!(stored.is_agent_visible(), "history stays visible ({label})");
        }
    }

    Ok(())
}
