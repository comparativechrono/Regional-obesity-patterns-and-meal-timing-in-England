"""Optional step 0: re-download the inputs that were fetched from public APIs.

The snapshots used in the paper are already in data/raw/; this script documents exactly how
they were obtained and can be used to verify them (default) or refresh them (--overwrite).
Other inputs were downloaded manually from the web pages listed in data/README.md.

0. OHID Fingertips NCMP indicators for upper-tier LAs (area type 502), all years, used for 2025/26.
   Open Government Licence v3.0. Note: Fingertips data are revised from time to time, so a fresh
   download may differ from the snapshot.
1. Census 2021 table RM033 "Ethnic group of dependent child by sex" (ONS, via Nomis dataset
   NM_2133_1), local authorities: county / unitary (as of April 2023). Open Government Licence v3.0.
2. Counties and Unitary Authorities (December 2023) Boundaries UK BUC (ultra-generalised, clipped),
   England only, from the ONS Open Geography Portal. Open Government Licence v3.0;
   contains OS data (c) Crown copyright and database right 2023. Used for drawing maps only.
"""
import sys
import urllib.request

from common import BOUNDARIES, CENSUS_ETHNICITY, FINGERTIPS_IDS, INDICATORS, RAW

SOURCES = {
    **{RAW / "fingertips_api" / INDICATORS[k][0]:
       f"https://fingertips.phe.org.uk/api/all_data/csv/by_indicator_id?indicator_ids={i}"
       "&child_area_type_id=502&parent_area_type_id=15" for k, i in FINGERTIPS_IDS.items()},
    CENSUS_ETHNICITY: (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2133_1.data.csv?date=latest&geography=TYPE423"
        "&c2021_eth_20=0,1001,1,2,3,4,5,1002,6,7,8,1003,1004,13,14,15,16,17,1005&c_sex=0&measures=20100"
        "&select=date_name,geography_name,geography_code,c2021_eth_20,c2021_eth_20_name,c_sex_name,"
        "measures_name,obs_value"),
    BOUNDARIES: (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BUC/FeatureServer/0/query"
        "?where=CTYUA23CD%20LIKE%20%27E%25%27&outFields=CTYUA23CD,CTYUA23NM&outSR=4326&f=geojson"),
}

if __name__ == "__main__":
    for dest, url in SOURCES.items():
        target = dest if "--overwrite" in sys.argv else dest.with_name(dest.stem + ".downloaded" + dest.suffix)
        urllib.request.urlretrieve(url, target)
        msg = "" if target == dest else f"; identical to snapshot: {target.read_bytes() == dest.read_bytes()}"
        print(f"saved {target.name}{msg}")
