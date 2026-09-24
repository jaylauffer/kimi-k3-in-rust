use std::process::Command;

#[test]
fn help_and_empty_invocation_do_not_load_a_model() {
    for args in [vec![], vec!["--help"], vec!["-h"]] {
        let out = Command::new(env!("CARGO_BIN_EXE_k3"))
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success());
        let text = String::from_utf8(out.stdout).unwrap();
        for required in ["--chat", "--gen", "--max-context", "/continue", "example:"] {
            assert!(text.contains(required));
        }
        assert!(out.stderr.is_empty());
    }
}

#[test]
fn invalid_options_fail_before_model_loading_with_a_help_hint() {
    for args in [
        vec!["missing-model", "--chat", "--prompt", "hi"],
        vec!["missing-model", "--chat", "--layers", "1"],
        vec!["missing-model", "--cache-gb", "NaN"],
        vec!["missing-model", "--cache-gb", "-1"],
        vec!["missing-model", "--gen", "0"],
        vec!["missing-model", "--gen"],
        vec!["missing-model", "--ring-slots", "0"],
        vec!["missing-model", "--max-context", "0"],
        vec!["missing-model", "--unknown"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_k3"))
            .args(args)
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        let text = String::from_utf8(out.stderr).unwrap();
        assert!(text.contains("run --help"), "{text}");
        assert!(
            !text.contains("cannot read"),
            "validation should precede model I/O"
        );
    }
}
