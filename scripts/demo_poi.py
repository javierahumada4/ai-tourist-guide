import sys
from ai_guide.pois import load_pois
from ai_guide.guide import make_guide_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo_poi.py <poi_id>")
        sys.exit(1)

    pois = load_pois("data/pois")
    poi_id = sys.argv[1]

    if poi_id not in pois:
        print("Available POIs:")
        for k in pois:
            print(" -", k)
        sys.exit(1)

    print(make_guide_text(pois[poi_id]))
