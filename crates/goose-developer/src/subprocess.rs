//! The part of goose's `subprocess.rs` the shell tool uses.

pub trait SubprocessExt {
    fn set_no_window(&mut self) -> &mut Self;
}

impl SubprocessExt for tokio::process::Command {
    fn set_no_window(&mut self) -> &mut Self {
        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW_FLAG: u32 = 0x08000000;
            self.creation_flags(CREATE_NO_WINDOW_FLAG);
        }
        self
    }
}
