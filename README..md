## Problem Objective :

The problem statement is perceived as a multi-class classification problem where we need to predict and recommend `Course` to be taken up by the student provided we have their `gender`, `stream`, `subject` and `marks` informations beforehand. The solution thus designed is simple, concise and demonstrated a lucid solution to perform `Course recommendation` for students based on the dataset provided.

## Solution Approach

### Training

1. Since it is a classical problem of `multi-class classification`, I have used basic `ensemble model` of 4 classifiers with voting mechanism to train the model using sklearn pipelines.
2. `sklearn-pipelines` helps to acheive better code quality in less time and is comparably more manageable, deployment-friendly and maintainable as compared to other approaches of development. This is chosen to ensure development of production ready code that is readily testable and deployed to any cloud platform.
3. Also, upon performing basic data analysis, it seems `EDA(Exploratory Data Analysis)` and `Databases` would not be required for this sample data as it is not very complex to handle with standard data libraries.
4. Although there are many options to select `training algorithmn` for this dataset, but I preferred to use one that is having `simple feature map` and does not have high compute requirements or parameters during `fitting` and `hyperparameter-tuning` phase. Also, the one whose `gradient-descent` time is least among others.
5. Executing the end code is pretty easy, just run a simple `train.py` python script file to be run from a specified location and your training for course recommendation system will be completed.


### Testing/ Inferencing

1. For `inferencing`, I am created a `FastAPI` based webapp as `backend` in python with no `GUI` as expected from problem statement. This will create a `local-demon` of classifier server in `localhost` , where we can send sample `REST API` call via `curl` or `Postman` to get the predicted and recommended `course` as a response from this App server.
2. Once tested locally, I have created a `Dockerfile` as an approach to `package` and `ship` my code to anyone like Alef education Hiring managers , who3. can simple run the `Dockerfile` , build the images and run the Container App anytime and anywere with breeze.
3. Note that the docker approach is only to perform `inferencing` and not for `training` the course recommendation system


## Steps to run and test my solution

### Recommendation System - Model Training

1. Unzip the `.zip` file sent as solution and keep it inside any directoy of your choice.
2. As a basic disclaimer, here is the directory structure being followed

<code>
/code  (code script folder)<br /> 
---------- train.py <br />
/data  (dataset folder)<br />
---------- dataset_csv_as provided <br />
/libr  (utility helper files)<br />
---------- __init__.py <br />
---------- utils.py <br />
/model (trained model artifacts)<br />
---------- model_binary.dat.gz <br />
---------- model_target_map.pkl <br />
/test (sample test examples)<br />
---------- sample_inferencing_examples <br />
Dockerfile (Docker file for build and run docker App)<br />
main.py ( Main App file )<br /> 
README.md (This file)<br />
requirements.txt (package dependency file)<br />

</code>

3. Install the package needed for creating a virtual environment. 
`pip install virtualenv` and create one with `virtualenv venv`. Also, activate it with `source venv/bin/activate`
4. Install all the package dependencies by installing the `requirements.txt` file.
`pip install -r requirements.txt`
5. `pip freeze` to check you have everything istalled and wthout any `error`
For training the model, simply run `python code/train.py` from root directory of this unzipped folder.
6. This will train the `classifier` model and save the `model-file` and `encoding-mapping-file` to `model/` folder ocation
7. Once you have these two file , then you can be asured that training of model has finished now.


### Recommendation System - Model Inferencing (Local)
1. To do a local inferencing , simply run `uvicorn main:app` command from root directory of this unzipped folder.
2. This will create and run a server of uvicom with url `http://127.0.0.1:8000` or `http://localhost:8000` 
3. This will expose few endpoints `/docs`, `/modelinfo`, `apphealth`, and `predict` with the above url.
4. Visit `http://127.0.0.1:8000/docs` or `http://localhost:8000/docs` and you can see three of these endpoints to try out.
5. Insert your test values for inferencing like below by entering like this and hit `execute`.

```
{"gender_code" : "female",
 "stream_code" : "science",
 "subject_code" : "math",
 "marks" : 73,}
```
6. You will see the below output for recommende course.
```
{"label":"btech",
  "prediction":3}
```

### Recommendation System - Model Inferencing (DOCKER)

1. This is a docker approach, so please install docker first in your system using standard official links to install.
2. Up and open your docker compose application
3. Build the docker images required for this course recommendation engine by running the below command.
`docker build . -t ml_test_solution `
4. This will create and build a docker image with name `ml_test_solution` .
5. Run this image with the below command `docker run -p 8000:8000 ml_test_solution`
6. Use the curl scripts from `test/sample_api_call.txt` to do inferencing .

```
# POST method predict
curl -d '{"gender_code": "female", 
        "stream_code": "science", 
        "subject_code": "math", 
        "marks": 73}' \
     -H "Content-Type: application/json" \
     -XPOST http://0.0.0.0:8000/predict

# GET method info
curl -XGET http://localhost:8000/info

# GET method health
curl -XGET http://localhost:8000/apphealth

```
7. The results will be silimar to below snap

```
{"label":"btech",
  "prediction":3}
```


## Submission checklist

1. All code scripts in zip file submitted are fully executable
2. It covers all 4 business requirements from problem statement 
3. It covers all 4 functional requirements from problem statement 
4. It demonstrates clean coding and test driven development.
5. In organised format
6. A Dockerfile as needed for deployments and testing the solution
7. A `train.py` file for training the model locally.
8. A webapp `main.py` with helper and utility files in a proper folder structure to run webapp
9. Added code comments for better and faster understanding
10. Everythng is my original work is alinged with the instructioned in this document.
