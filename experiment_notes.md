# Experiment Notes — King County Spatial SHAP

## 1. K selection  
Figure 20, Figure 21  
Table: `KingCounty_final_K_selection_combined.csv`

Figure 20: CV error 
tested K = 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100  
lowest CV RMSE: K=15  
K=20: very close to the minimum CV (RMSE=1.01) 
larger K: CV error slowly increases  
too large K: less local  

Figure 21: SHAP stability  
small K values(5,10): not stable enough  
K=15: better, but min K was set as 20  
K=20: both CV rule and stability rule  
K=25, 30, 40 also pass, but less local than K=20  

Final choice: K=20  
Reason: still local, but not too unstable.


## 2. Main result for grade  
Figure 11  
Table: `spatial_shap_summary_statistics.csv`

Figure 11: Standard SHAP vs KNN Spatial SHAP for `grade`  
50 test houses   
diagonal line = Standard SHAP and Spatial SHAP give the same value  
many points away from the diagonal line: clear difference between global and local attribution  

Table result:  
mean absolute difference: about 62.6k dollars  
sign-change rate: 18%  

Main result:  
`grade` attribution changes after using local spatial background  
for some houses, local background changes how `grade` is interpreted


## 3. Sensitivity around K=20  
Figure 13  
Tables:  
`KingCounty_KNN_sensitivity_top5_summary.csv`  
`KingCounty_KNN_correlation_matrix_grade.csv`

Figure 13: heatmap for `grade`  
compare K = 10, 15, 20, 25, 30  
K=20 vs K=25: very high correlation  
K=20 vs K=30: still highly similar  
K=15 vs K=20: also stable  

Table result:  
top-five-feature summary confirms the same pattern  
K=20, 25, 30: high Pearson / Spearman correlation  
high sign agreement

Main result:  
K=20 result is stable  
not caused by one random K choice  
changing K a little around 20 does not change the result much


## 4. Location effect  
Figure 14  
Table: `KingCounty_location_effect_summary_300_houses.csv`

Figure 14: location effect map  
300 test houses  
location effect = local baseline - global baseline  
not a normal feature SHAP value  
shows how much the local background changes the baseline  

Table result:  
mean absolute location effect: about 148k dollars  
location effect: has large range  
some local baselines much higher/lower than global baseline   

Main result:  
local baseline is very different from global baseline  
housing data has strong spatial structure  
one global background is too simple for this dataset


## 5. Standard SHAP map for grade  
Figure 17  
Table: `KingCounty_standard_vs_spatial_map_grade_300.csv`

Figure 17: Standard SHAP map for `grade`  
300 test houses  
global-background SHAP 
same global reference for all houses, used as the baseline map  

Main result:  
shows how `grade` is explained under standard SHAP   
not the strongest result alone


## 6. Spatial SHAP map for grade  
Figure 18  
Table: `KingCounty_standard_vs_spatial_map_grade_300.csv`

Figure 18: KNN Spatial SHAP map for `grade`  
300 test houses  
K=20 local background  
local reference = 20 nearest training houses

Main result:  
`grade` attribution changes after using local background  
interpretation: local-neighbourhood based


## 7. Difference map  
Figure 19  
Table: `KingCounty_map_comparison_summary_grade_300.csv`

Figure 19: Spatial SHAP minus Standard SHAP  
300 test houses  
direct difference between local and global explanation   
most important map: show where local background changes the explanation most  

Table result:  
mean absolute difference: about 65.2k dollars  
sign-change rate: about 15.3%  

Main result:  
difference is not small and random
local-vs-global SHAP difference has spatial pattern  
confirms Figure 11 result on a larger 300-house map


## 8. Decomposition check  
Figure 15  
Tables:  
`KingCounty_decomposition_summary_300_houses.csv`  
`KingCounty_decomposition_check_300_houses.csv`

Figure 15: decomposition error  
300 test houses  
check whether the decomposition works 
decomposition idea: global explanation can be separated into location effect + local feature attribution  

Table result:  
mean absolute decomposition error: about 0.003  
RMSE: very close to zero  
error is basically zero compared with house-price scale  

Main result:  
decomposition works  
implementation looks numerically reliable  
good validation result, not just a visual result


## Overall result

Final K = 20.  
it keeps local background both local and also stable

Main feature example = `grade`.  
Standard SHAP and Spatial SHAP give clearly different attributions for `grade`.

50-house result:  
mean absolute difference: around 62.6k dollars  
sign-change rate: 18%

300-house map result:  
mean absolute difference: around 65.2k dollars  
sign-change rate: 15.3%

Location effect:  
mean absolute location effect: around 148k dollars  
local baseline: very different from global baseline

Decomposition:  
error almost zero, implementation reliable

Order:
1. K selection: Figure 20 + Figure 21  
2. main comparison: Figure 11  
3. sensitivity: Figure 13  
4. location effect: Figure 14  
5. maps: Figure 17, 18, 19  
6. validation: Figure 15