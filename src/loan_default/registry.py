"""Promote a registered model and export it for the prediction service.

    python -m loan_default.train --register                      # train, log, add a new version
    python -m loan_default.registry promote --version 3         # point the "champion" alias at version 3
    python -m loan_default.registry promote --version 2 --force # roll back to a weaker version
    python -m loan_default.registry export --out-dir served      # write the champion as model.joblib + metadata.json

The service reads a directory with model.joblib and metadata.json (MODEL_DIR), so an
exported champion can be served without rebuilding the image:

    docker run -v "$PWD/served:/app/models:ro" -p 8000:8000 loan-default
"""

import argparse
import json
from pathlib import Path

import joblib

from loan_default.tracking import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI

DEFAULT_ALIAS = "champion"


def version_test_auc(name: str, version: str, tracking_uri: str) -> float:
    """Test ROC-AUC recorded in the metadata of one registered version."""
    import mlflow
    from mlflow import MlflowClient

    run_id = MlflowClient(tracking_uri).get_model_version(name, version).run_id
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="metadata.json", tracking_uri=tracking_uri)
    return float(json.loads(Path(path).read_text())["metrics"]["test"]["roc_auc"])


def promote(
    version: int,
    name: str = DEFAULT_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> None:
    """Point `alias` at one version, unless it scores below the version the alias points at now.

    `force` skips the comparison, for a deliberate rollback.
    """
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    if not force:
        # read the alias table instead of catching any error: a failed lookup must not skip the comparison
        current = client.get_registered_model(name).aliases.get(alias)
        if current is not None and current != str(version):
            candidate_auc = version_test_auc(name, str(version), tracking_uri)
            current_auc = version_test_auc(name, current, tracking_uri)
            if candidate_auc < current_auc:
                raise ValueError(
                    f"version {version} has test ROC-AUC {candidate_auc:.4f}, below {alias} version {current} "
                    f"({current_auc:.4f}); use --force to promote it anyway"
                )
    client.set_registered_model_alias(name, alias, str(version))


RELEASES = "releases"
CURRENT = "current"


def export(
    out_dir: Path, name: str = DEFAULT_MODEL_NAME, alias: str = DEFAULT_ALIAS, tracking_uri: str = DEFAULT_TRACKING_URI
) -> int:
    """Publish the model version behind `alias`, with its training metadata, as one release; return the version.

    The alias is resolved once, and the model and metadata both come from that version. They are written
    into a new directory `out_dir/releases/v<version>-<id>/` that is never modified afterwards; then the
    symbolic link `out_dir/current` is switched to it in one atomic rename. A reader that resolves
    `current` once (as the service does) always gets a model and a cutoff from the same release, even if
    an export is running or was interrupted. A running service caches its release: restart it to serve
    the new one.
    """
    import os
    import tempfile

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    version = MlflowClient(tracking_uri).get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    releases = out_dir / RELEASES
    releases.mkdir(parents=True, exist_ok=True)
    release = Path(tempfile.mkdtemp(prefix=f"v{version.version}-", dir=releases))
    with tempfile.TemporaryDirectory() as staging:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="metadata.json", dst_path=staging, tracking_uri=tracking_uri
        )
        metadata = json.loads(Path(downloaded).read_text())
    metadata["registered_model"] = {"name": name, "version": int(version.version), "alias": alias}
    joblib.dump(model, release / "model.joblib")
    (release / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    release.chmod(0o755)
    link = out_dir / f".{CURRENT}.{os.getpid()}.tmp"
    link.unlink(missing_ok=True)
    link.symlink_to(Path(RELEASES) / release.name, target_is_directory=True)
    link.replace(out_dir / CURRENT)  # atomic switch: readers see the old release or the new one, never a mix
    return int(version.version)


def main(argv: list[str] | None = None) -> None:
    """Promote or export from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    commands = parser.add_subparsers(dest="command", required=True)
    promote_parser = commands.add_parser("promote", help="point the alias at a version")
    promote_parser.add_argument("--version", type=int, required=True)
    promote_parser.add_argument("--force", action="store_true", help="skip the comparison with the current version")
    commands.add_parser("export", help="write the aliased version to a directory").add_argument(
        "--out-dir", type=Path, default=Path("served")
    )
    args = parser.parse_args(argv)

    if args.command == "promote":
        promote(args.version, args.name, args.alias, args.tracking_uri, force=args.force)
        print(f"{args.name}@{args.alias} -> version {args.version}")
    else:
        version = export(args.out_dir, args.name, args.alias, args.tracking_uri)
        print(f"Exported {args.name} version {version} ({args.alias}) to {args.out_dir}")


if __name__ == "__main__":
    main()
