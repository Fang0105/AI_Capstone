# Description
The training dataset is obtained from comments of a video on YouTube. The topic of the video is the election debate between Joe Biden and Donald Trump. These comments are fetched from Google API, filling the id of the video and save into a csv file. Then remove emojis in the comments and get rid of those comments that are not in English. Finally, use Textblob model to label each comment, 1 for Positive, 0 for Neutral, -1 for Negative. This dataset is for training and testing during the training process. Besides the comments from the debate, I downloaded other comments from a video whose topic is the discussion of the debate between NewJeans, a Korean girl group, and its companies, Hybe and Ador in court. This dataset is for testing the model only.
|                 | Training | Testing |
|-----------------|----------|---------|
| Election Debate | 55877    | 13970   |
| Court Debate    | x        | 123     |

|                 | Positive | Neutral | Negative |
|-----------------|----------|---------|----------|
| Election Debate | 27917    | 25364   | 16566    |
