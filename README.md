# Web Application for Intracranial Hemorrhage Detection

 The objective of this project is to create a web tool for neuro-specialists to assist them in brain hemorrhage detection
 and localization task. An abstract of this project has been submitted to the `Society for Imaging Informatics in Medicine 2019`.


## Table of Contents

*   [Directory Layout](#directory-layout)
*   [How to Run](#how-to-run)
    *   [Requirements](#requirements)
    *   [Steps](#steps)
*   [Sample DICOMS](#sample_dicoms)
*   [Demo](#demo)
*   [Courtesy](#courtesy)
*   [Contributors](#contributors)
*   [Supervisors](#supervisors)


## Directory layout

```
├── app.py
├── demo
├── Sample DICOM Images
├── gradcam
├── inference_rsna.py
├── model
├── models
│   └── fold0_ep3.pt
├── README.md
├── requirements.txt
├── static
├── templates
├── test.py
├── uploads
└── utils.py
```


### Requirements
- **Python packages**: Available in `requirements.txt`
- **[ngrok](https://dashboard.ngrok.com/auth)**
  
### Steps

- **Set Up ngrok**:
  - Open an account using your mail address [here](https://ngrok.com/)
  - Find your `authtoken` [here](https://dashboard.ngrok.com/auth).
  - Make `ngrok` executable on your system and run the following command:
  
  
  ```
  $ ./ngrok authtoken <YOUR_AUTH_TOKEN>
  ``` 
- **Running Web server**: 
    - Go to `Intracranial_Hemorrhage_Web_App` directory.
    - Run `python app.py`. The web server will start running on `localhost:9999`
    - Open another terminal and run `ngrok http 9999`. After a couple of seconds, it will generate a 
    texts that will look like this
    
    
    
~~~
ngrok by @inconshreveable                                       (Ctrl+C to quit)
                                                                                
Session Status                online                                            
Account                       <your_name> (Plan: Free)                               
Update                        update available (version 2.3.35, Ctrl-U to update
Version                       2.3.34                                            
Region                        United States (us)                                
Web Interface                 http://127.0.0.1:4040                             
Forwarding                    http://********.ngrok.io -> http://localhost:9999 
Forwarding                    https://********.ngrok.io -> http://localhost:9999
                                                                                
Connections                   ttl     opn     rt1     rt5     p50     p90       
                              243     0       0.00    0.00    0.01    3.81    
~~~
   - Copy the link after `Forwarding` (http://********.ngrok.io). Now anyone can access the web server from
   this link. 

## Sample DICOMS
`Sample DICOM Images` directory contains 12 images. Their labels are available in the `Ground_Truth.csv` files.


## Demo
- [Step step tutorial for using the web application](shorturl.at/fjvS6)
- [Demo Video](https://drive.google.com/file/d/1aMc6MvjvYXU76Nkai5OKM5I1km2lmJgC/view?usp=sharing) 


## Courtesy
 - [11th place solution](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage) of the RSNA Hemorrhage Detection Competition.
 - [Grad-CAM ++ Pytorch](https://github.com/vickyliin/gradcam_plus_plus-pytorch) 
 
 
## Contributors
 - **[Tahsin Mostafiz](https://github.com/tahsin314)** 
 - **[Shajib Ghosh](https://github.com/ShajibGhosh)**


## Supervisors
-  **[Dr. Taufiq Hasan](http://bme.buet.ac.bd/?teams=dr-taufiq-hasan)**
-  **[Dr. Paul Naggy](https://www.hopkinsmedicine.org/profiles/results/directory/profile/2936930/paul-nagy)**