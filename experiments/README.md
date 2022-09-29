# organoid-segmentation-ML

APOC:

*  <b>Experiment 07</b> - APOC ProbabilityMapper FeaturesSet0
*  <b>Experiment 07-1</b> - APOC ProbabilityMapper FeaturesSet1
*  <b>Experiment 07-2</b> - APOC ProbabilityMapper FeaturesSet2
*  <b>Experiment 07-3</b> - APOC ProbabilityMapper FeaturesSet3
*  <b>Experiment 07-4</b> - APOC ProbabilityMapper FeaturesSet4
*  <b>Experiment 07-5</b> - APOC ProbabilityMapper FeaturesSet5
*  <b>Experiment 07-6</b> - APOC ProbabilityMapper FeaturesSet6
*  <b>Experiment 07-7</b> - APOC ProbabilityMapper FeaturesSet7
*  <b>Experiment 07-8</b> - APOC ProbabilityMapper FeaturesSet8
*  <b>Experiment 07-9</b> - APOC ProbabilityMapper FeaturesSet9
*  <b>Experiment 07-10</b> - APOC ProbabilityMapper FeaturesSet10
*  <b>Experiment 07-11</b> - APOC ProbabilityMapper FeaturesSet11
*  <b>Experiment 07-12</b> - APOC ProbabilityMapper FeaturesSet12
*  <b>Experiment 07-13</b> - APOC ProbabilityMapper FeaturesSet13
*  <b>Experiment 07-14</b> - APOC ProbabilityMapper FeaturesSet14
*  <b>Experiment 07-15</b> - APOC ProbabilityMapper FeaturesSet15
*  <b>Experiment 07-16</b> - APOC ProbabilityMapper FeaturesSet16
*  <b>Experiment 07-17</b> - APOC ProbabilityMapper FeaturesSet17
*  <b>Experiment 07-18</b> - APOC ProbabilityMapper FeaturesSet18
*  <b>Experiment 07-19</b> - APOC ProbabilityMapper FeaturesSet19
*  <b>Experiment 07-20</b> - APOC ProbabilityMapper FeaturesSet20
*  <b>Experiment 07-neg</b> - APOC ProbabilityMapper only on original data, no features used

VollSeg:

*  <b>Experiment 02</b> - prediction with pretrained Vollseg UNET model
*  <b>Experiment 04</b> - custom trained UNET for VollSeg 
  
Semantic Segmentation:
*  <b>Experiment 03</b> - semantic segmentation with custom trained UNET model
  
  *  <b>Experiment 03-1</b> - semantic segmentation with custom trained UNET model, Plantseg (anisotropic) dataset. Trained custom UNET with binary labels, and then tried Vollseg. Did not work.
  *  <b>Experiment 03-2</b> - semantic segmentation with custom trained UNET model, Plantseg (anisotropic) dataset. Trained custom UNET with binary labels, and then tried Vollseg. Did not work.
  *  <b>Experiment 03-3</b> - semantic segmentation with custom trained UNET model, Plantseg (anisotropic) dataset. Trained custom UNET with binary labels, and then tried Vollseg. In some places really visible edges of patches. Not sufficient patches overlap.
  *  <b>Experiment 03-4</b> - semantic segmentation with custom trained UNET model, Plantseg (anisotropic) dataset. Trained custom UNET for Plantseg prediction postprocessing with masking. In some places really visible edges of patches. Not sufficient patches overlap.
  *  <b>Experiment 03-5</b> - semantic segmentation with custom trained UNET model, Plantseg (isotropic) dataset. Downscaled data (0.5, 0.5, 0.5), patches (64, 224, 224), epochs 60, steps/epoch 291. Big GPU

* StarDist:

  *  <b>Experiment 01</b> - initial StarDist training (StarDist dataset)
  *  <b>Experiment 10</b> - StarDist training with raw data, downsampled 0.8 (StarDist dataset)
  *  <b>Experiment 11</b> - StarDist training with raw data, downsampled 0.8 (StarDist dataset)

StarDist + APOC:

*  <b>Experiment 09</b> - StarDist custom trained on probability maps from APOC as raw data (did not work because random forest classifier prediction did not work on normalized images)
*  <b>Experiment 12</b> - raw data: prob maps with F1 (downsampled 0.8, no normalization)
*  <b>Experiment 12-1</b> - raw data: prob maps with F2 (downsampled 0.8, no normalization)
*  <b>Experiment 12-2</b> - raw data: prob maps with F3 (downsampled 0.8, no normalization)
*  <b>Experiment 12-3</b> - raw data: prob maps with F4 (downsampled 0.8, no normalization)
*  <b>Experiment 12-4</b> - raw data: prob maps with F16 (downsampled 0.8, no normalization)
*  <b>Experiment 12-5</b> - raw data: prob maps with F20 (downsampled 0.8, no normalization)

Plantseg:

*  <b>Experiment 05</b> - prediction with pretrained Plantseg UNET model
*  <b>Experiment 13</b> - Plantseg pretrained, prediction on isotropic data (takes way too long -> memory problems)
  
  *  <b>Experiment 13-1</b> - Plantseg pretrained (prediction on anisotropic data, test w/o norm dataset)
  *  <b>Experiment 13-2</b> - Plantseg pretrained (prediction on anisotropic data, test w/o normalization dataset)
  *  <b>Experiment 13-3</b> - Plantseg pretrained (prediction on anisotropic data, plantseg dataset)
  
*  <b>Experiment 13-3 + 03-4</b> - Postprocessing Plantseg results from exp13-3 with masks from UNET prediction (trained for semantic segmentation)
*  <b>Experiment 14</b> - Training custom UNET for Plantseg.
*  <b>Experiment 14-1</b> - Continuing training UNET for Plantseg with pretrained weights (light-sheet model)
  
Watershed:

*  <b>Experiment 16</b> - Thresholded local minima seeded watershed
*  <b>Experiment 17</b> - Seeded watershed using StarDist prediction as seeds

