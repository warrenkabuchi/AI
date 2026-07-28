from copilot.cli import build_parser


def test_defaults():
    args = build_parser().parse_args([])
    assert args.mode == "copilot"
    assert args.source == "default"
    assert args.display == "terminal"
    assert args.model == "base.en"
    assert args.script is None


def test_teleprompter_args():
    args = build_parser().parse_args(
        ["--mode", "teleprompter", "--script", "talk.md", "--display", "overlay"])
    assert args.mode == "teleprompter"
    assert args.script == "talk.md"
