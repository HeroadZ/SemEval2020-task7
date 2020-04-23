# SemEval2020-task7
This repository provides source code and detail explanation for WUY team in SemEval 2020 Task 7.

## how to use it

### 1. install dependencies
```bash
pip install -r requirements.txt
```

### 2. choose your operations
In the main file, you can make two operations, training and showing results.  
  
I saved all the pretrained model in my local environments. If you want to train by yourself, use
```bash
python main.py -t
```
        
I saved the reults in the pickle file, so you can directly show results without training. Showing of subtask A, use
```bash
python main.py -a
```  
           
show result of subtask B, use
```bash
python main.py -b
```  
          
           
show result of both subtasks, use
```bash
python main.py -s
```
