#! /usr/bin/env python3
import argparse
import os
import subprocess
import uuid
from typing import List

DIM = 3
TARGET_SIZES = [1_000, 10_000, 100_000]
REORDERINGS = ["none", "rcm", "amd", "metis"]
SYM_TYPES = ["symmetric", "near_symmetric", "general"]
DRY_RUN = False


def size_factory(ts):
    return round(ts ** (1. / DIM))


def size_thyra(ts):
    return round((ts / 3) ** (1. / DIM))


def build_filename(**kwargs):
    return "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))


def parameter_list_factory(*, reordering="none", sym_type="general"):
    return f"""<ParameterList name="Solver Factory Example">
    <Parameter name="SolverType" type="string" value="GinkgoSolver"/>
    <ParameterList name="GinkgoSolver">
        <Parameter name="Reordering" type="string" value="{reordering}"></Parameter>
        <Parameter name="SymbolicType" type="string" value="{sym_type}"></Parameter>
    </ParameterList>
</ParameterList>
    """


def parameter_list_thyra(*, reordering="none", sym_type="general"):
    return f"""<ParameterList name="Thyra Example">
    <Parameter name="Linear Solver Type" type="string" value="Belos"/>
    <ParameterList name="Linear Solver Types">
        <ParameterList name="Belos">
            <Parameter name="Solver Type" type="string" value="Block GMRES"/>
            <ParameterList name="Solver Types">
                <ParameterList name="Block GMRES">
                    <Parameter name="PreconditionerPosition" type="string" value="left"/>
                    <Parameter name="Block Size" type="int" value="1"/>
                    <Parameter name="Convergence Tolerance" type="double" value="1e-4"/>
                    <Parameter name="Maximum Iterations" type="int" value="100"/>
                    <Parameter name="Output Frequency" type="int" value="1"/>
                    <Parameter name="Show Maximum Residual Norm Only" type="bool" value="1"/>
                </ParameterList>
            </ParameterList>
        </ParameterList>
    </ParameterList>
    <Parameter name="Preconditioner Type" type="string" value="FROSch"/>
    <ParameterList name="Preconditioner Types">
        <ParameterList name="FROSch">
            <Parameter name="FROSch Preconditioner Type" type="string" value="GDSWPreconditioner"/>

            <ParameterList name="AlgebraicOverlappingOperator">
                <ParameterList name="Solver">
                    <Parameter name="SolverType" type="string" value="GinkgoSolver"/>
                    <ParameterList name="GinkgoSolver">
                        <Parameter name="Reordering" type="string" value="{reordering}"/>
                        <Parameter name="SymbolicType" type="string" value="{sym_type}"/>
                    </ParameterList>
                </ParameterList>
            </ParameterList>

            <ParameterList name="GDSWCoarseOperator">
                <ParameterList name="Blocks">
                    <ParameterList name="1">
                        <Parameter name="Use For Coarse Space" type="bool" value="true"/>
                        <Parameter name="Rotations" type="bool" value="true"/>
                    </ParameterList>
                </ParameterList>

                <ParameterList name="ExtensionSolver">
                    <Parameter name="SolverType" type="string" value="GinkgoSolver"/>
                    <ParameterList name="GinkgoSolver">
                        <Parameter name="Reordering" type="string" value="{reordering}"/>
                        <Parameter name="SymbolicType" type="string" value="{sym_type}"/>
                    </ParameterList>
                </ParameterList>

                <ParameterList name="Distribution">
                    <Parameter name="Type" type="string" value="linear"/>
                    <Parameter name="NumProcs" type="int" value="1"/>
                    <Parameter name="Factor" type="double" value="1.0"/>
                    <Parameter name="GatheringSteps" type="int" value="1"/>
                    <ParameterList name="Gathering Communication">
                        <Parameter name="Send type" type="string" value="Send"/>
                    </ParameterList>
                </ParameterList>

                <ParameterList name="CoarseSolver">
                    <Parameter name="SolverType" type="string" value="GinkgoSolver"/>
                    <ParameterList name="GinkgoSolver">
                        <Parameter name="Reordering" type="string" value="{reordering}"/>
                        <Parameter name="SymbolicType" type="string" value="{sym_type}"/>
                    </ParameterList>
                </ParameterList>
            </ParameterList>
        </ParameterList>
    </ParameterList>
</ParameterList>
"""


def run(*, size, parameter_list, prefix, exe, args):
    for ts in TARGET_SIZES:
        for reordering in args.reorderings:
            if reordering == "none" and ts == 1_000_000:
                continue

            for sym_type in args.sym_types:
                if sym_type == "general" and ts == 1_000_000:
                    continue

                actual_size = size(ts)

                tmp_file_name = uuid.uuid4().hex[:6]
                with open(tmp_file_name, "w") as tmp_file:
                    tmp_file.write(parameter_list(reordering=reordering, sym_type=sym_type))

                run_args = ["mpirun", "-n", f"{args.n}", f"{prefix}/{exe}", f"--M={actual_size}", f"--PLIST={tmp_file_name}",
                        "--USETPETRA", f"--DIM={DIM}"]

                print("Running: ", run_args)
                with open(tmp_file_name, "r") as file:
                    print(file.read())

                if not DRY_RUN:
                    try:
                        cp = subprocess.run(run_args, capture_output=True, universal_newlines=True, check=True)

                        with open(build_filename(app=args.app, np=args.n, size=actual_size, reordering=reordering,
                                                 sym_type=sym_type),
                                  "w") as file:
                            file.write(cp.stdout)

                        print(" " * 4, "Success")
                    except subprocess.CalledProcessError as e:
                        print(e.stderr)

                os.remove(tmp_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("--reorderings", type=str, nargs='+', default=REORDERINGS, choices=REORDERINGS)
    parser.add_argument("--sym-types", type=str, nargs='+', default=SYM_TYPES, choices=SYM_TYPES)
    parser.add_argument("app", type=str, choices=["factory", "thyra"])
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run

    if args.factory:
        prefix = "packages/shylu/shylu_dd/frosch/test/SolverFactory"
        exe = "ShyLU_DDFROSch_solverfactory.exe"
        if args.n == 0:
            args.n = 1
        run(size=size_factory, parameter_list=parameter_list_factory, prefix=prefix, exe=exe, args=args)

    if args.thyra:
        prefix = "packages/shylu/shylu_dd/frosch/test/Thyra_Xpetra_Elasticity"
        exe = "ShyLU_DDFROSch_thyra_xpetra_elasticity.exe"
        if args.n == 0:
            args.n = 8

        run(size=size_thyra, parameter_list=parameter_list_thyra, prefix=prefix, exe=exe, args=args)


if __name__ == "__main__":
    main()
