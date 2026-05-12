import json
import os


def main():
    report = {
        "GUROBI_HOME": os.environ.get("GUROBI_HOME"),
        "GRB_LICENSE_FILE": os.environ.get("GRB_LICENSE_FILE"),
        "gurobipy_import": False,
        "gurobi_runtime_ok": False,
        "error": None,
    }
    try:
        import gurobipy as gp  # type: ignore

        report["gurobipy_import"] = True
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY, name="x")
        model.update()
        report["gurobi_runtime_ok"] = True
    except Exception as exc:
        report["error"] = str(exc)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["gurobi_runtime_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
