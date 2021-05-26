#!/usr/bin/env python3
"""
This is a waLBerla parameter file that tests (almost) all parameter combinations for GPU communication.
Build waLBerla with -DWALBERLA_BUILD_WITH_PYTHON=1  then run e.g.
 ./UniformGridBenchmarkGPU_AA_trt simulation_setup/benchmark_configs.py

Look at the end of the file to select the benchmark to run
"""

import os
import waLBerla as wlb
from waLBerla.tools.config import block_decomposition
from waLBerla.tools.sqlitedb import sequenceValuesToScalars, checkAndUpdateSchema, storeSingle
import sys
import sqlite3

# Number of time steps run for a workload of 128^3 per GPU
# if double as many cells are on the GPU, half as many time steps are run etc.
# increase this to get more reliable measurements
TIME_STEPS_FOR_128_BLOCK = 500
DB_FILE = "gpu_benchmark.sqlite3"


def num_time_steps(block_size):
    cells = block_size[0] * block_size[1] * block_size[2]
    time_steps = (128 ** 3 / cells) * TIME_STEPS_FOR_128_BLOCK
    return int(time_steps)


class Scenario:
    def __init__(self, cells_per_block=(256, 128, 128), periodic=(1, 1, 1), cuda_blocks=(256, 1, 1),
                 timesteps=None, time_step_strategy="normal", omega=1.8, cuda_enabled_mpi=False,
                 inner_outer_split=(1, 1, 1), warmup_steps=5, outer_iterations=3, init_shear_flow=False,
                 communication_scheme="UniformGPUScheme_Baseline"):

        self.blocks = block_decomposition(wlb.mpi.numProcesses())

        self.cells_per_block = cells_per_block
        self.periodic = periodic

        self.time_step_strategy = time_step_strategy
        self.omega = omega
        self.timesteps = timesteps if timesteps else num_time_steps(cells_per_block)
        self.cuda_enabled_mpi = cuda_enabled_mpi
        self.inner_outer_split = inner_outer_split
        self.init_shear_flow = init_shear_flow
        self.warmup_steps = warmup_steps
        self.outer_iterations = outer_iterations
        self.cuda_blocks = cuda_blocks
        self.communication_scheme = communication_scheme

        self.vtk_write_frequency = 0

        self.config_dict = self.config(print_dict=False)

    @wlb.member_callback
    def config(self, print_dict=True):
        from pprint import pformat
        config_dict = {
            'DomainSetup': {
                'blocks': self.blocks,
                'cellsPerBlock': self.cells_per_block,
                'periodic': self.periodic,
            },
            'Parameters': {
                'omega': self.omega,
                'cudaEnabledMPI': self.cuda_enabled_mpi,
                'warmupSteps': self.warmup_steps,
                'outerIterations': self.outer_iterations,
                'timeStepStrategy': self.time_step_strategy,
                'timesteps': self.timesteps,
                'initShearFlow': self.init_shear_flow,
                'gpuBlockSize': self.cuda_blocks,
                'communicationScheme': self.communication_scheme,
                'innerOuterSplit': self.inner_outer_split,
                'vtkWriteFrequency': self.vtk_write_frequency
            }
        }
        if print_dict:
            wlb.log_info_on_root("Scenario:\n" + pformat(config_dict))
        return config_dict

    @wlb.member_callback
    def results_callback(self, **kwargs):
        data = {}
        data.update(self.config_dict['Parameters'])
        data.update(self.config_dict['DomainSetup'])
        data.update(kwargs)
        data['executable'] = sys.argv[0]
        data['compile_flags'] = wlb.build_info.compiler_flags
        data['walberla_version'] = wlb.build_info.version
        data['build_machine'] = wlb.build_info.build_machine
        sequenceValuesToScalars(data)

        result = data
        sequenceValuesToScalars(result)
        num_tries = 4
        # check multiple times e.g. may fail when multiple benchmark processes are running
        for num_try in range(num_tries):
            try:
                checkAndUpdateSchema(result, "runs", DB_FILE)
                storeSingle(result, "runs", DB_FILE)
                break
            except sqlite3.OperationalError as e:
                wlb.log_warning(f"Sqlite DB writing failed: try {num_try + 1}/{num_tries}  {str(e)}")


# -------------------------------------- Profiling -----------------------------------
def profiling():
    """Tests different communication overlapping strategies"""
    wlb.log_info_on_root("Running 2 timesteps for profiling")
    wlb.log_info_on_root("")

    scenarios = wlb.ScenarioManager()
    cells = (256, 256, 256)
    cuda_enabled_mpi = False

    scenarios.add(Scenario(cells_per_block=cells, time_step_strategy='kernelOnly',
                           communication_scheme='UniformGPUScheme_Baseline',
                           inner_outer_split=(1, 1, 1), timesteps=2, cuda_enabled_mpi=cuda_enabled_mpi,
                           outer_iterations=1, warmup_steps=0))


# -------------------------------------- Functions trying different parameter sets -----------------------------------
def benchmark_all():
    """Tests different communication overlapping strategies"""
    wlb.log_info_on_root("Running different communication overlap strategies")
    wlb.log_info_on_root("")

    scenarios = wlb.ScenarioManager()
    cell_sizes = [(i, i, i) for i in (64, 128, 192, 256, 320, 384)]
    cuda_enabled_mpi = False

    inner_outer_splits = [(1, 1, 1), (4, 1, 1), (8, 1, 1), (16, 1, 1), (32, 1, 1),
                          (4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1),
                          (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]

    cuda_blocks = [(32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1), (512, 1, 1),
                   (32, 2, 1), (64, 2, 1), (128, 2, 1), (256, 2, 1),
                   (32, 4, 1), (64, 4, 1), (128, 4, 1),
                   (32, 8, 1), (64, 8, 1),
                   (32, 16, 1)]

    # 'GPUPackInfo_Baseline', 'GPUPackInfo_Streams'
    for cells in cell_sizes:
        for cuda_block_size in cuda_blocks:
            for comm_strategy in ['UniformGPUScheme_Baseline', 'UniformGPUScheme_Memcpy']:
                # no overlap
                scenarios.add(Scenario(time_step_strategy='noOverlap',
                                       cuda_blocks=cuda_block_size,
                                       communication_scheme=comm_strategy,
                                       inner_outer_split=(1, 1, 1),
                                       cuda_enabled_mpi=cuda_enabled_mpi))

                # overlap
                for overlap_strategy in ['simpleOverlap', 'complexOverlap']:
                    for inner_outer_split in inner_outer_splits:
                        if any([inner_outer_split[i] * 2 >= cells[i] for i in range(len(inner_outer_split))]):
                            continue
                        scenario = Scenario(time_step_strategy=overlap_strategy,
                                            cuda_blocks=cuda_block_size,
                                            communication_scheme=comm_strategy,
                                            inner_outer_split=inner_outer_split,
                                            cuda_enabled_mpi=cuda_enabled_mpi)
                        scenarios.add(scenario)


def overlap_benchmark():
    """Tests different communication overlapping strategies"""
    wlb.log_info_on_root("Running different communication overlap strategies")
    wlb.log_info_on_root("")

    scenarios = wlb.ScenarioManager()
    cells = (256, 256, 256)
    cuda_enabled_mpi = False
    inner_outer_splits = [(1, 1, 1), (4, 1, 1), (8, 1, 1), (16, 1, 1), (32, 1, 1),
                          (4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1),
                          (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]

    # 'GPUPackInfo_Baseline', 'GPUPackInfo_Streams'
    for comm_strategy in ['UniformGPUScheme_Baseline', 'UniformGPUScheme_Memcpy']:
        # no overlap
        scenarios.add(Scenario(time_step_strategy='noOverlap',
                               communication_scheme=comm_strategy,
                               inner_outer_split=(1, 1, 1),
                               cuda_enabled_mpi=cuda_enabled_mpi))

        # overlap
        for overlap_strategy in ['simpleOverlap', 'complexOverlap']:
            for inner_outer_split in inner_outer_splits:
                scenario = Scenario(time_step_strategy=overlap_strategy,
                                    communication_scheme=comm_strategy,
                                    inner_outer_split=inner_outer_split,
                                    cuda_enabled_mpi=cuda_enabled_mpi)
                scenarios.add(scenario)


def communication_compare():
    """Tests different communication strategies"""
    wlb.log_info_on_root("Running benchmarks to compare communication strategies")
    wlb.log_info_on_root("")

    scenarios = wlb.ScenarioManager()
    cuda_enabled_mpi = False
    for cells in [(384, 384, 384)]:
        for comm_strategy in ['GPUPackInfo_Baseline', 'GPUPackInfo_Streams',
                              'UniformGPUScheme_Baseline', 'UniformGPUScheme_Memcpy',
                              'MPIDatatypes', 'MPIDatatypesFull']:

            sc = Scenario(cells_per_block=cells,
                          cuda_blocks=(128, 1, 1),
                          time_step_strategy='noOverlap',
                          communication_scheme=comm_strategy,
                          cuda_enabled_mpi=cuda_enabled_mpi)
            scenarios.add(sc)
            for inner_outer_split in [(4, 1, 1), (8, 1, 1), (16, 1, 1), (32, 1, 1)]:
                # ensure that the inner part of the domain is still large enough
                if any([inner_outer_split[i] * 2 >= cells[i] for i in range(len(inner_outer_split))]):
                    continue
                sc = Scenario(cells_per_block=cells,
                              cuda_blocks=(128, 1, 1),
                              time_step_strategy='simpleOverlap',
                              inner_outer_split=inner_outer_split,
                              communication_scheme=comm_strategy,
                              cuda_enabled_mpi=cuda_enabled_mpi)
                scenarios.add(sc)


def single_gpu_benchmark():
    """Benchmarks only the LBM compute kernel"""
    wlb.log_info_on_root("Running single GPU benchmarks")
    wlb.log_info_on_root("")

    scenarios = wlb.ScenarioManager()
    cell_sizes = [(i, i, i) for i in (64, 128, 192, 256, 320, 384)]
    cuda_blocks = [(32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1), (512, 1, 1),
                   (32, 2, 1), (64, 2, 1), (128, 2, 1), (256, 2, 1),
                   (32, 4, 1), (64, 4, 1), (128, 4, 1),
                   (32, 8, 1), (64, 8, 1),
                   (32, 16, 1)]

    cuda_enabled_mpi = False
    for cells in cell_sizes:
        for cuda_block_size in cuda_blocks:
            for time_step_strategy in ['kernelOnly', 'noOverlap']:
                scenario = Scenario(cells_per_block=cells,
                                    cuda_blocks=cuda_block_size,
                                    time_step_strategy=time_step_strategy,
                                    cuda_enabled_mpi=cuda_enabled_mpi)
                scenarios.add(scenario)


# -------------------------------------- Optional job script generation for PizDaint ---------------------------------


job_script_header = """
#!/bin/bash -l
#SBATCH --job-name=scaling
#SBATCH --time=0:30:00
#SBATCH --nodes={nodes}
#SBATCH -o out_scaling_{nodes}_%j.txt
#SBATCH -e err_scaling_{nodes}_%j.txt
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=d105

cd {folder}

source ~/env.sh

module load daint-gpu
module load craype-accel-nvidia60
export MPICH_RDMA_ENABLED_CUDA=1  # allow GPU-GPU data transfer
export CRAY_CUDA_MPS=1            # allow GPU sharing
export MPICH_G2G_PIPELINE=256     # adapt maximum number of concurrent in-flight messages

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

export MPICH_RANK_REORDER_METHOD=3
export PMI_MMAP_SYNC_WAIT_TIME=300


# grid_order -R -H -c 1,1,8 -g 16,16,8

ulimit -c 0
"""

job_script_exe_part = """

export WALBERLA_SCENARIO_IDX=0
while srun -n {nodes} ./{app} {config}
do
 ((WALBERLA_SCENARIO_IDX++))
done
"""

all_executables = ('UniformGridBenchmarkGPU_mrt_d3q27',
                   'UniformGridBenchmarkGPU_smagorinsky_d3q27',
                   'UniformGridBenchmarkGPU_cumulant'
                   'UniformGridBenchmarkGPU_cumulant_d3q27')


def generate_jobscripts(exe_names=all_executables):
    for node_count in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2400]:
        folder_name = "scaling_{:04d}".format(node_count)
        os.makedirs(folder_name, exist_ok=True)

        # run grid_order
        import subprocess
        decomposition = block_decomposition(node_count)
        decomposition_str = ",".join(str(e) for e in decomposition)
        subprocess.check_call(['grid_order', '-R', '-H', '-g', decomposition_str])

        job_script = job_script_header.format(nodes=node_count, folder=os.path.join(os.getcwd(), folder_name))
        for exe in exe_names:
            job_script += job_script_exe_part.format(app="../" + exe, nodes=node_count,
                                                     config='../communication_compare.py')

        with open(os.path.join(folder_name, 'job.sh'), 'w') as f:
            f.write(job_script)


if __name__ == '__main__':
    print("Called without waLBerla - generating job scripts for PizDaint")
    generate_jobscripts()
else:
    wlb.log_info_on_root("Batch run of benchmark scenarios, saving result to {}".format(DB_FILE))
    # Select the benchmark you want to run
    # single_gpu_benchmark()  # test different cell sizes and cuda block layouts. Suitable for single node runs.
    # profiling()  # very small run with only two timesteps. Suitable for profiling.
    # benchmark_all()  # benchmark all different variants. This benchmark takes a long time
    communication_compare()  # benchmarks different communication routines, with and without overlap
    # overlap_benchmark()  # benchmarks different communication overlap options
