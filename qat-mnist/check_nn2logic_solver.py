import json
import os
import sys


def configure_windows_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_dirs = [
        os.path.join(repo_root, "nn2logic"),
        os.path.join(repo_root, "tools", "vcpkg", "installed", "x64-windows", "bin"),
    ]

    try:
        import gurobipy as gp  # type: ignore

        candidate_dirs.append(os.path.dirname(gp.__file__))
    except Exception:
        pass

    gurobi_home = os.environ.get("GUROBI_HOME")
    if gurobi_home:
        candidate_dirs.append(os.path.join(gurobi_home, "bin"))

    for dll_dir in candidate_dirs:
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)


def main():
    report = {
        "nn2logic_import": False,
        "input_encoder_ok": False,
        "error": None,
        "gurobi_home": os.environ.get("GUROBI_HOME"),
        "grb_license_file": os.environ.get("GRB_LICENSE_FILE"),
    }

    try:
        configure_windows_dll_dirs()
        try:
            from nn2logic import InputEncoder  # type: ignore
        except ImportError:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            vendored = os.path.join(repo_root, "nn2logic")
            if vendored not in sys.path:
                sys.path.insert(0, vendored)
            from nn2logic import InputEncoder  # type: ignore

        report["nn2logic_import"] = True
        enc = InputEncoder()
        enc.registerInt("x_0", 1)
        enc.update()
        report["input_encoder_ok"] = True
    except Exception as exc:
        report["error"] = str(exc)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["input_encoder_ok"]:
        msg = report["error"] or ""
        if "10009" in msg and "PIP license" in msg:
            print(
                "\n[accion] Licencia actual de gurobi es tipo PIP (solo gurobipy). "
                "nn2logic usa API C++ y requiere licencia completa (academic/commercial).",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
