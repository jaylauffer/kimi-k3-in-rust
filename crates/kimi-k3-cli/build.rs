//! macOS asks a program for its reasons before granting microphone or speech-recognition
//! access (`--voice`), and stops one that has none. A command-line binary carries them in
//! an Info.plist embedded in its `__TEXT,__info_plist` section.
fn main() {
    println!("cargo:rerun-if-changed=kimi-Info.plist");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
        let plist = std::path::Path::new(&dir).join("kimi-Info.plist");
        println!(
            "cargo:rustc-link-arg-bin=k3=-Wl,-sectcreate,__TEXT,__info_plist,{}",
            plist.display()
        );
    }
}
