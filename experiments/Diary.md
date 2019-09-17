## Diary

Add sections in reverse Chronological order

1. Mask
2. Multi-layer
3. heat map

Tuesday, September 10, 2019
1. Add a new feature 'area_ratio' into features. Regenerate feature vectors for prior experiments. Repeat the random patch experiment.
2. Color windows to reflect detection scores. Mark cells meanwhile.

Monday, September 9, 2019
1. Finish the process for 3D region shift. It takes more than 1000 hours for one instance.
2. Able to mark cells selected by different feature value ranges with multi-colors. Test on structures both far apart and close together. 

Thursday, September 5, 2019
1. Get the region shift move by grids across the x-y-z space.
2. Export the heat maps for shift to pdf. Change the style of half-way value contours.
3. Test the cells marking codes on sections with multiple structures. Just mark the most important feature.

Wednesday, September 4, 2019
1. Use the adaptive thresholds to generate masks.
2. Manually select value ranges for important features after sorting features according to feature importance. Change the max depth of boost trees to be 1.
3. Mark cells whose feature values are within the selected ranges.

Tuesday, September 3, 2019
1. Draw heat maps to show dependence of detection scores on translation. Mark the maximum and half-way contours.
2. Test the codes to mark cells whose feature values are within a small range of selected thresholds for important features.

Monday, September 2, 2019
1. Get the region shift move by grids across x-y plane. It takes about 3.5 hours for 10 EC2 instances.

Friday, August 30, 2019
1. Use the 'get_score' function to get feature importance of each feature. Sort features by 'total gain' values.
2. Mark the cells whose feature values are chosen for tree boosters. Warm colors are features with great importance.

Thursday, August 29, 2019
1. Change the step size of region shift from 30 microns to 20 microns and the range of shift from 600 microns to 800 microns.
2. Draw shift curves for new shift results. Mark the maximum and half-way value of curves.


Wednesday, August 28, 2019
1. Mark the cells whose feature values show big difference in positive and negative CDF curves for each feature. Each color represents one feature.
2. Look through the GUI part of MouseBrainAtlas_dev.

Tuesday, August 27, 2019
1. Get initial results for shift ready. It takes about 40 minutes for 10 EC2 instances.
2. Draw shift curves for each structure.
2. Get the codes for hsv images marking cells run locally successfully.

Monday, August 26, 2019
1. Correct the cells collection part for region shift and hsv images.
2. Get the codes for region shift run locally successfully.
3. Test the codes for hsv images.

Thursday-Friday, August 22-23, 2019

Sick leave

Wednesday, August 21, 2019

Progress:
1. Get the pipeline for region shift using the sqlite database run locally successfully. --Shape_shift_V2.py
2. Debug the codes for region shift on EC2 instances.
3. Look through the GUI part of MouseBrainAtlas_dev.

Tuesday, August 20, 2019

Progress:
1. Debug the codes for region shift both locally and on EC2 instances.
2. Establish a second pipeline for region shift using the sqlite database. --Shape_shift_V2.py

Problems:
1. Unstable connection from EC2 instances to S3 resulted missing files during the upload process.

Monday, August 19, 2019

Progress:
1. Add the part enabling shift in z direction. --Shape_shift.py
2. Test on EC2 instances.

Saturday, August 17, 2019

Progress:
1. All sections have been completed for the test. It takes about 400 hours for a single instance.

Downsample Images:
Google Drive: https://drive.google.com/drive/folders/1gJaPJfPpTNimbBC0mkd1Sq7P_3_0F724?usp=sharing

Friday, August 16, 2019

Progress:
1. Test the whole process with stride of 50um. (112 sections completed)
2. Debug the codes both locally and on EC2 instances.

Downsample Images:
Google Drive: https://drive.google.com/drive/folders/1gJaPJfPpTNimbBC0mkd1Sq7P_3_0F724?usp=sharing

Notes: 
1. The following structures are recommended due to enough contour images: 
5N, 7N, 7nn, DC, IC, LRt, PBG, Pn, SC, SNC, SNR, Sp5C, Sp5I, Sp5O, VCA, VCP, VLL.
2. Images with '_contour' in their names have annotation contours.


Thursday, August 15, 2019

Progress:
1. Divide the dictionary storing feature vectors into tiny files to avoid memory issues.
2. Debug the codes both locally and on EC2 instances.
3. Test the whole process with larger stride of 50um. (Still running, 300 hours estimated for single instance)

Problems:
1. Part of EC2 instances may be terminated automatically (Server.SpotInstanceTermination: may due to price). I will discuss this with Julaiti tomorrow.

Wednesday, August 14, 2019

Progress:
1. Correct the downsample part for score maps.
2. Debug the codes on EC2 instances.

Problems:
1. For the stride of 30um, it will takes about 5 hours for one section and more than 1000 hours in total.
Wall time is much greater than cpu time.
2. Codes stop half way without error reports. It may due to memory issues.

Tuesday, August 13, 2019

Progress:
1. Get the codes for scoremaps run on multiple EC2 instances successfully.
2. Change the stride of sliding windows to 30um. This may extend computation time by 3 times.

Problems:
1. Lack some preprocessed files to convert annotations to ones applicable to prep2 images. I will discuss this with Alex.

**Structures with good ROC: 3N, 4N, 7nn, 12N, Amb, AP, LC, PBG, Pn, RMC, Tz, VCA, VLL

Monday, August 12, 2019

Progress:
1. Get the codes for generating scoremaps run locally successfully.
2. Add the part to draw annotations and test successfully on the brain MD594.

**Yoav:** Kui, can you put here pointers to (some of) the generated scoremaps?
s3: https://s3.console.aws.amazon.com/s3/buckets/mousebrainatlas-data/CSHL_scoremaps_new/?region=us-west-1&tab=overview
Google Drive: https://drive.google.com/drive/folders/1YGXhe4mnfsY1RRxqGFMTR_RpKrr3NkXV?usp=sharing

Notes: 
1. Here are some results for MD594. Annotations only appear in the images that this structure shows in this section.
2. Current results of VCA are recommended to review cause they include one whole side of this structure.
3. It takes about 50 minutes for one section (28 scoremaps) on EC2. 
4. the sliding window is 103.04um (224 pixels) on the edge with a stride of 51.52um (112 pixles). But the resolution of results seems low. I will ask Alex about the window size of previous results.
5. Sigmoid is used to normalize the scoremaps in order to make scores have same gray values across sections. 


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
