# video_moment_detection

1. Clone the github repository

2. We test the code with:

   pytorch1.2.0

   python3.7

   cuda9.0 or higher

3. To get the MAP values for all Six domains, you need:

    download the feature files 'ytbData.pkl':
    https://drive.google.com/open?id=1yA49xsZb8VaEqn6QTqXMNPhi_EOfnydo
    
    dowload youtube highlight datasets: 
    git clone 'https://github.com/aliensunmin/DomainSpecificHighlight.git'
    
    download models:
    https://drive.google.com/open?id=1x_Ys2WMRKzIMlP528GYJEn0fv0EznDJs

4. RUN:  
   ./get_map_for_allDomains.sh
   
   output:
   
Loading test data
predicting scores for snippets...
calculate MAP...
dog_MAP:0.6332961132363135
Loading test data
predicting scores for snippets...
calculate MAP...
gymnastics_MAP:0.8251755624960364
Loading test data
predicting scores for snippets...
calculate MAP...
parkour_MAP:0.6929103789873162
Loading test data
predicting scores for snippets...
calculate MAP...
skating_MAP:0.5299066519046772
Loading test data
predicting scores for snippets...
calculate MAP...
skiing_MAP:0.7445942742030888
Loading test data
predicting scores for snippets...
calculate MAP...
surfing_MAP:0.7932121734343105




