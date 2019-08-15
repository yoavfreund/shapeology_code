## Diary

Add sections in reverse Chronological order

Tuesday, Auguest 13, 2019
Progress:
1. Get the codes for scoremaps run on multiple EC2 instances successfully.
2. Change the stride of sliding windows to 30um. This may extend computation time by 3 times.

Problems:
1. Lack some preprocessed files to convert annotations to ones applicable to prep2 images. I will discuss this with Alex.

**Structures with good ROC: 3N, 4N, 7nn, 12N, Amb, AP, LC, PBG, Pn, RMC, Tz, VCA, VLL

Monday, Auguest 12, 2019
Progress:
1. Get the codes for generating scoremaps run locally successfully.
2. Add the part to draw annotations and test successfully on the brain MD594.

**Yoav:** Kui, can you put here pointers to (some of) the generated scoremaps?
s3: https://s3.console.aws.amazon.com/s3/buckets/mousebrainatlas-data/CSHL_scoremaps_new/?region=us-west-1&tab=overview
Google Drive: https://drive.google.com/drive/folders/1YGXhe4mnfsY1RRxqGFMTR_RpKrr3NkXV?usp=sharing

Note: Here are some results for MD594. Annotations only appear in the images that this structure shows in this section. 
Current results of VCA are recommended to review cause they include one whole side of this structure.
It takes about 50 minutes for one section (28 scoremaps) on EC2. 
And the sliding window is 103.04um (224 pixels) on the edge with a stride of 51.52um (112 pixles). But the resolution of results seems low. I will ask Alex about the window size of previous results.
Sigmoid is used to normalize the scoremaps in order to make scores have same gray values across sections. 


Problems:
1. Annotations for brains except MD585, 589 and 594 are not consistent with the prep2 images. 
Need to realign these annotations. I will discuss this with Alex.


Friday, July 12, 2019
1. Show hsv images classified by good/bad, structures, train/test sequentially. Put on Google drive.
Further step: turn good/bad into both good, both bad, and train good but test bad. Divide images of paired structure into two sides.
2. Show time each step took during the process.
Further step: move load DM part to init. If possible, try to improve DM source codes.

Next step:
1. Perform cell extraction on the whole section with tiles with no overlap. 
2. Store cells in pandas table instead of dictionary. (Suggest: SQLite)
Each row contains x,y,section relative to the whole brain, feature vector

3. Calculate scores for shifted structures.
for each structure, we have positive and negative volumes 
for a given x,y,z shift, compute the region-based features and a score using xgboost
calculate score in correct place with different dx, dy, dz