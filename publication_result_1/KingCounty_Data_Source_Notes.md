# King County Housing Dataset — Data Source and Variable Descriptions

## 1. Data Source

**Dataset:** House Sales in King County, USA  
**Observations:** 21,613  
**Period:** May 2014 – May 2015  
**Location:** King County, Washington, USA

**Kaggle — Dataset Download:**  
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

**King County Department of Assessments — Official Property Records:**  
https://info.kingcounty.gov/assessor/DataDownload/default.aspx

**GeoDa Data and Lab — Variable Documentation:**  
https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/

**R Package — geodaData: kchomesale:**  
https://spatialanalysis.github.io/geodaData/reference/kchomesale.html

The dataset used in this study was downloaded from Kaggle. The King County Department of Assessments provides official property sales and residential building records containing corresponding property identifiers, transaction information, and housing characteristics. Preliminary comparisons identified matching property identifiers, sale dates, and prices between selected Kaggle observations and official records.

## 2. Variable Descriptions

Variable descriptions are based on GeoDa documentation and the King County Department of Assessments' official record descriptions.

| Variable | Description | Unit / Values |
|---|---|---|
| `id` | Property identification number | Identifier |
| `date` | Date sold | Date |
| `price` | House sale price | USD |
| `bedrooms` | Number of bedrooms | Count |
| `bathrooms` | Number of bathrooms | Count |
| `sqft_living` | Interior living area | Square feet |
| `sqft_lot` | Size of the land lot | Square feet |
| `floors` | Number of floors | Count |
| `waterfront` | Whether the property has a waterfront | 1 = Yes; 0 = No |
| `view` | Rating of the property's view | 0–4 |
| `condition` | Overall condition of the house | 1–5 |
| `grade` | Construction quality based on materials and workmanship | Construction grade |
| `sqft_above` | Above-ground floor area | Square feet |
| `sqft_basement` | Below-ground floor area | Square feet |
| `yr_built` | Year built | Year |
| `yr_renovated` | Year renovated; 0 if never renovated | Year |
| `zipcode` | Property ZIP code | 5-digit ZIP code |
| `lat` | Latitude | Decimal degrees |
| `long` | Longitude | Decimal degrees |
| `sqft_living15` | Average interior living area of the closest 15 houses | Square feet |
| `sqft_lot15` | Average land lot area of the closest 15 houses | Square feet |

**Notes:** The descriptions of `sqft_living15` and `sqft_lot15` follow GeoDa documentation. The official King County Lookup records provide additional definitions for `grade` and `condition`.