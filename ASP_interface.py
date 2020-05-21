import subprocess
from pathlib import Path
def solve(asp_lines):
    asp_lines = asp_lines + ['%%%%%%%%%%%%', 'display', '%%%%%%%%%%%%', 'value.'] 
    f=open("solver_files/sudokuSolver_sample.sp", "r+")  
    content = f.read()
    f.close()
    f=open('solver_files/sudokuSolver.sp','w+')
    f.write(content + '\n'.join(asp_lines))
    f.close()
    solver_compiler_path = Path(__file__).cwd().as_posix()+'/solver_files/sparc.jar'
    solver_path = Path(__file__).cwd().as_posix()+'/solver_files/sudokuSolver.sp'
    answerSet = subprocess.check_output(' '.join(['java','-jar', solver_compiler_path, solver_path, '-A']),shell=True).decode("utf-8")
    chosenAnswer = answerSet.strip().split('\n\n')[0]
    entries = chosenAnswer.strip('{}').split(', ')
    return entries