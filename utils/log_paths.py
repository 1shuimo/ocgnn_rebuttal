from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT.parent / "rebuttal_log"


def add_log_subdir_argument(parser, default_subdir):
    parser.add_argument("--log_subdir", type=str, default=default_subdir)


def get_log_file(args, filename):
    log_dir = DEFAULT_LOG_ROOT / args.log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / filename)
