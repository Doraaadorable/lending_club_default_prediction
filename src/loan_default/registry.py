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
    from mlflow.exceptions import MlflowException

    client = MlflowClient(tracking_uri)
    if not force:
        try:
            current = client.get_model_version_by_alias(name, alias).version
        except MlflowException:
            current = None
        if current is not None and current != str(version):
            candidate_auc = version_test_auc(name, str(version), tracking_uri)
            current_auc = version_test_auc(name, current, tracking_uri)
            if candidate_auc < current_auc:
                raise ValueError(
                    f"version {version} has test ROC-AUC {candidate_auc:.4f}, below {alias} version {current} "
                    f"({current_auc:.4f}); use --force to promote it anyway"
                )
    client.set_registered_model_alias(name, alias, str(version))


def export(
    out_dir: Path, name: str = DEFAULT_MODEL_NAME, alias: str = DEFAULT_ALIAS, tracking_uri: str = DEFAULT_TRACKING_URI
) -> int:
    """Write the model version behind `alias` and its training metadata to `out_dir`; return the version."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    version = MlflowClient(tracking_uri).get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="metadata.json", dst_path=str(out_dir), tracking_uri=tracking_uri
    )
    metadata = json.loads(Path(downloaded).read_text())
    metadata["registered_model"] = {"name": name, "version": int(version.version), "alias": alias}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
