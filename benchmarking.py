#! /usr/bin/env python3
import os
import subprocess
import uuid

DIM = 3
TARGET_SIZES = [1_000, 10_000, 100_000]
REORDERINGS = ["none", "rcm", "amd"]
SYM_TYPE = ["symmetric", "near_symmetric", "general"]

PREFIX = "packages/shylu/shylu_dd/frosch/test/SolverFactory"
EXE = "ShyLU_DDFROSch_solverfactory.exe"


def size(ts):
    return round(ts ** (1. / DIM))


def build_filename(**kwargs):
    return "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))


def parameter_list(*, reordering="none", sym_type="general"):
    return f"""<ParameterList name="Solver Factory Example">
    <Parameter name="SolverType" type="string" value="GinkgoSolver"/>
    <ParameterList name="GinkgoSolver">
        <Parameter name="Reordering" type="string" value="{reordering}"></Parameter>
        <Parameter name="SymbolicType" type="string" value="{sym_type}"></Parameter>
    </ParameterList>
</ParameterList>
    """


for ts in TARGET_SIZES:
    for reordering in REORDERINGS:
        if reordering == "none" and ts == 1_000_000:
            continue

        for sym_type in SYM_TYPE:
            if sym_type == "general" and ts == 1_000_000:
                continue

            actual_size = size(ts)

            tmp_file_name = uuid.uuid4().hex[:6]
            with open(tmp_file_name, "w") as tmp_file:
                tmp_file.write(parameter_list(reordering=reordering, sym_type=sym_type))

            args = [f"{PREFIX}/{EXE}", f"--M={actual_size}", f"--PLIST={tmp_file_name}", "--USETPETRA", f"--DIM={DIM}"]

            try:
                print("Running: ", args)
                with open(tmp_file_name, "r") as file:
                    print(file.read())

                for _ in range(TARGET_SIZES[-1] // ts):
                    cp = subprocess.run(args, capture_output=True, universal_newlines=True, check=True)

                with open(build_filename(size=actual_size, reordering=reordering, sym_type=sym_type), "w") as file:
                    file.write(cp.stdout)

                print(" " * 4, "Success")
            except subprocess.CalledProcessError as e:
                print(e.stderr)
            finally:
                os.remove(tmp_file_name)
