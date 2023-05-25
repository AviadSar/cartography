import sys
import os
import time
import copy

while True:
    jobs_to_submit = []
    with open('/cs/labs/roys/aviadsa/cartography/slurm_configs/submitted_jobs.txt', 'r') as submitted_jobs:
        submitted_jobs = submitted_jobs.readlines()
        new_submitted_jobs = []
        os.system('sacct > slurm_configs/sacct_results.txt')
        with open('/cs/labs/roys/aviadsa/cartography/slurm_configs/sacct_results.txt', 'r') as sacct_results:
            sacct_results = sacct_results.readlines()[2:]
            for job in submitted_jobs:
                split_job = job.split()
                job_id, job_command = split_job[3], ' '.join(split_job[4:])
                found = False
                running = False
                completed = True
                pending = False
                for sacct_result in sacct_results:
                    result_split = sacct_result.split()
                    result_id = result_split[0].split('.')[0]
                    result_status = result_split[-2]
                    print(result_status)
                    if job_id in result_id:
                        found = True
                        if result_status == 'PENDING':
                            pending = True
                        if result_status == 'RUNNING':
                            running = True
                        if result_status != 'COMPLETED':
                            completed = False
                if running or pending:
                    new_submitted_jobs.append(job)
                elif not found:
                    job_output_dir = job_command.split()[1].replace('slurm_configs', 'outputs').replace('.sh', '')
                    if os.path.exists(job_output_dir):
                        for file in os.listdir(job_output_dir):
                            if file == 'done.txt':
                                break
                        else:
                            jobs_to_submit.append(job_command)
                elif found and not completed:
                    jobs_to_submit.append(job_command)

    with open('/cs/labs/roys/aviadsa/cartography/slurm_configs/submitted_jobs.txt', 'w') as submitted_jobs:
        submitted_jobs.writelines(new_submitted_jobs)

    print(jobs_to_submit)
    for job_command in jobs_to_submit:
        os.system('{0} >> slurm_configs/submitted_jobs.txt  && truncate -s-1 slurm_configs/submitted_jobs.txt && echo -n " {0}\n" >> slurm_configs/submitted_jobs.txt'.format(job_command))

    time.sleep(600)