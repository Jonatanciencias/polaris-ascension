"""
Wildlife Dataset Downloader for Radeon RX 580 AI Framework

Downloads wildlife/conservation datasets for testing inference performance.
Supports multiple sources including Colombia-specific data.

Datasets supported:
- iNaturalist Colombia: Native species (birds, mammals, reptiles)
- Snapshot Serengeti: African wildlife (baseline comparison)
- Custom: Local camera trap images

Usage:
    # Download Colombia wildlife data
    python scripts/download_wildlife_dataset.py --region colombia --species all --num-images 500
    
    # Download specific species
    python scripts/download_wildlife_dataset.py --region colombia --species "jaguar,spectacled_bear,tapir" --num-images 200
    
    # Download Snapshot Serengeti for comparison
    python scripts/download_wildlife_dataset.py --region serengeti --num-images 1000
"""

import argparse
import json
import requests
from pathlib import Path
from typing import List, Dict
import time
from urllib.parse import urlencode
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class WildlifeDatasetDownloader:
    """Download wildlife images from various sources."""
    
    def __init__(self, output_dir: str = "data/wildlife"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colombia-specific species (iconic and endangered)
        self.colombia_species = {
            'jaguar': {'scientific': 'Panthera onca', 'common': 'Jaguar', 'endangered': True},
            'spectacled_bear': {'scientific': 'Tremarctos ornatus', 'common': 'Oso de anteojos', 'endangered': True},
            'mountain_tapir': {'scientific': 'Tapirus pinchaque', 'common': 'Danta de montaña', 'endangered': True},
            'ocelot': {'scientific': 'Leopardus pardalis', 'common': 'Ocelote', 'endangered': False},
            'puma': {'scientific': 'Puma concolor', 'common': 'Puma', 'endangered': False},
            'white_tailed_deer': {'scientific': 'Odocoileus virginianus', 'common': 'Venado cola blanca', 'endangered': False},
            'capybara': {'scientific': 'Hydrochoerus hydrochaeris', 'common': 'Chigüiro', 'endangered': False},
            'howler_monkey': {'scientific': 'Alouatta seniculus', 'common': 'Mono aullador', 'endangered': False},
            'harpy_eagle': {'scientific': 'Harpia harpyja', 'common': 'Águila arpía', 'endangered': True},
            'king_vulture': {'scientific': 'Sarcoramphus papa', 'common': 'Rey de los gallinazos', 'endangered': False},
        }
        
    def download_inaturalist_colombia(self, species: List[str], num_images: int = 500):
        """
        Download observations from iNaturalist for Colombian wildlife.
        
        Uses the iNaturalist API to download real wildlife images from Colombia.
        All images are Creative Commons licensed and research-grade quality.
        
        Args:
            species: List of species common names or 'all'
            num_images: Maximum images to download per species
        """
        print("\n" + "="*70)
        print("🇨🇴 DOWNLOADING COLOMBIA WILDLIFE DATA (iNaturalist)")
        print("="*70)
        
        if 'all' in species:
            species_list = list(self.colombia_species.keys())
        else:
            species_list = species
        
        print(f"\n📋 Target species: {len(species_list)}")
        for sp in species_list:
            if sp in self.colombia_species:
                info = self.colombia_species[sp]
                status = "🔴 ENDANGERED" if info['endangered'] else "🟢 Stable"
                print(f"   • {info['common']} ({info['scientific']}) - {status}")
        
        print(f"\n🌐 Connecting to iNaturalist API...")
        print(f"   Place ID: 7827 (Colombia)")
        print(f"   Quality: research-grade only")
        print(f"   License: Creative Commons (downloadable)")
        print()
        
        # Create output directory
        colombia_dir = self.output_dir / "colombia"
        colombia_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'region': 'colombia',
            'source': 'iNaturalist',
            'species_count': len(species_list),
            'species': [],
            'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': 0
        }
        
        base_url = "https://api.inaturalist.org/v1/observations"
        total_downloaded = 0
        
        for sp in species_list:
            if sp not in self.colombia_species:
                print(f"⚠️  Unknown species: {sp}, skipping...")
                continue
            
            species_info = self.colombia_species[sp]
            print(f"\n📥 {species_info['common']} ({species_info['scientific']})")
            
            # Create species directory
            species_dir = colombia_dir / sp
            species_dir.mkdir(exist_ok=True)
            
            # Query iNaturalist API
            params = {
                'taxon_name': species_info['scientific'],
                'place_id': 7827,  # Colombia
                'quality_grade': 'research',
                'photos': 'true',
                'per_page': min(num_images, 200),  # API limit
                'order': 'desc',
                'order_by': 'created_at'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                observations = data.get('results', [])
                print(f"   Found {len(observations)} observations")
                
                if not observations:
                    print(f"   ⚠️  No observations found, skipping...")
                    continue
                
                # Download images
                downloaded_count = 0
                species_metadata = {
                    'common_name': species_info['common'],
                    'scientific_name': species_info['scientific'],
                    'endangered': species_info['endangered'],
                    'images': []
                }
                
                for idx, obs in enumerate(observations[:num_images]):
                    if 'photos' not in obs or not obs['photos']:
                        continue
                    
                    photo = obs['photos'][0]  # Use first photo
                    photo_url = photo.get('url', '').replace('square', 'medium')
                    
                    if not photo_url:
                        continue
                    
                    # Download image
                    img_filename = f"{sp}_{obs['id']}.jpg"
                    img_path = species_dir / img_filename
                    
                    if img_path.exists():
                        continue
                    
                    try:
                        print(f"   ⬇️  Downloading {idx+1}/{len(observations)}...", end='\r')
                        
                        img_response = requests.get(photo_url, timeout=15)
                        img_response.raise_for_status()
                        
                        img_path.write_bytes(img_response.content)
                        downloaded_count += 1
                        
                        # Store metadata
                        species_metadata['images'].append({
                            'filename': img_filename,
                            'observation_id': obs['id'],
                            'observer': obs.get('user', {}).get('login', 'unknown'),
                            'observed_on': obs.get('observed_on_string', 'unknown'),
                            'location': obs.get('place_guess', 'Colombia'),
                            'license': photo.get('license_code', 'unknown'),
                            'url': f"https://www.inaturalist.org/observations/{obs['id']}"
                        })
                        
                        time.sleep(0.5)  # Be respectful to API
                        
                    except Exception as e:
                        print(f"   ⚠️  Error downloading image {idx+1}: {e}")
                        continue
                
                print(f"   ✅ Downloaded {downloaded_count} images")
                total_downloaded += downloaded_count
                
                metadata['species'].append(species_metadata)
                
                # Save species metadata
                species_meta_path = species_dir / "metadata.json"
                with open(species_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(species_metadata, f, indent=2, ensure_ascii=False)
                
            except requests.RequestException as e:
                print(f"   ❌ API Error: {e}")
                continue
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                continue
        
        metadata['total_images'] = total_downloaded
        
        # Save overall metadata
        meta_path = colombia_dir / "dataset_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("✅ DOWNLOAD COMPLETE")
        print("="*70)
        print(f"📊 Total images downloaded: {total_downloaded}")
        print(f"📁 Location: {colombia_dir}")
        print(f"📋 Metadata: {meta_path}")
        print()
        print("💡 Next steps:")
        print(f"   1. Review images in: {colombia_dir}")
        print(f"   2. Use with: python examples/use_cases/wildlife_monitoring.py")
        print(f"   3. Train custom model on this dataset")
        print()
    
    def download_snapshot_serengeti_sample(self, num_images: int = 1000):
        """
        Download sample from Snapshot Serengeti for baseline comparison.
        
        This is one of the largest camera trap datasets with 2.65M images.
        We'll use this as a comparison baseline for our Colombia demo.
        
        Args:
            num_images: Number of images to download
        """
        print("\n" + "="*70)
        print("🦁 SNAPSHOT SERENGETI BASELINE DATASET")
        print("="*70)
        
        print(f"\n📊 Dataset info:")
        print(f"   Total images: 2.65 million")
        print(f"   Species: 48 (lions, elephants, zebras, etc.)")
        print(f"   Location: Serengeti National Park, Tanzania")
        print(f"   Use case: Baseline comparison for our Colombia demo")
        
        print(f"\n⚠️  This is a large dataset. For demo purposes:")
        print(f"   1. Visit: https://lila.science/datasets/snapshot-serengeti")
        print(f"   2. Download: SnapshotSerengetiS01.zip (~10GB for season 1)")
        print(f"   3. Extract to: {self.output_dir}/serengeti/")
        print(f"   4. We'll use a sample for benchmarking")
        
        # Create metadata
        metadata = {
            'region': 'serengeti',
            'source': 'LILA (Microsoft AI for Earth)',
            'species_count': 48,
            'total_images': 2650000,
            'sample_size': num_images,
            'use_case': 'Baseline comparison for RX 580 performance',
            'download_url': 'https://lila.science/datasets/snapshot-serengeti',
            'citation': 'Swanson et al. (2015). Snapshot Serengeti, high-frequency annotated camera trap images of 40 mammalian species in an African savanna'
        }
        
        serengeti_dir = self.output_dir / 'serengeti'
        serengeti_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = self.output_dir / 'serengeti_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Metadata created: {metadata_file}")
        
        return metadata
    
    def create_demo_dataset(self):
        """
        Create a small demo dataset with sample images for testing.
        
        Uses ImageNet classes that contain animals for immediate testing
        without large downloads.
        """
        print("\n" + "="*70)
        print("🎯 CREATING DEMO DATASET (ImageNet Wildlife)")
        print("="*70)
        
        # ImageNet classes that are wildlife
        wildlife_classes = {
            'n02119789': 'kit_fox',
            'n02120079': 'arctic_fox',
            'n02129165': 'lion',
            'n02129604': 'tiger',
            'n02130308': 'cheetah',
            'n02132136': 'brown_bear',
            'n02133161': 'american_black_bear',
            'n02134084': 'sloth_bear',
            'n02137549': 'mongoose',
            'n02317335': 'spider_monkey',
            'n02325366': 'wood_rabbit',
            'n02326432': 'hare',
            'n02391049': 'zebra',
            'n02395406': 'hog',
            'n02396427': 'wild_boar',
            'n02402425': 'gazelle',
            'n02403003': 'ox',
            'n02408429': 'water_buffalo',
            'n02410509': 'bison',
            'n02412080': 'ram',
            'n02415577': 'bighorn_sheep',
            'n02417914': 'ibex',
            'n02423022': 'antelope',
            'n02437312': 'impala',
            'n02437616': 'llama',
            'n02441942': 'weasel',
            'n02443114': 'polecat',
            'n02444819': 'otter',
            'n02445715': 'skunk',
            'n02447366': 'badger',
            'n02480495': 'orangutan',
            'n02480855': 'gorilla',
            'n02481823': 'chimpanzee',
            'n02483362': 'gibbon',
            'n02486410': 'baboon',
            'n02487347': 'macaque',
            'n02488702': 'colobus_monkey',
            'n02489166': 'proboscis_monkey',
            'n02490219': 'marmoset',
            'n02492035': 'capuchin_monkey',
            'n02492660': 'howler_monkey',  # ¡Presente en Colombia!
            'n02493509': 'titi_monkey',
            'n02493793': 'squirrel_monkey',
            'n02494079': 'spider_monkey',  # ¡Presente en Colombia!
        }
        
        demo_dir = self.output_dir / 'demo'
        demo_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'region': 'demo',
            'source': 'ImageNet (subset)',
            'species_count': len(wildlife_classes),
            'use_case': 'Quick demo without large downloads',
            'classes': wildlife_classes,
            'note': 'Use your own test images or download from iNaturalist'
        }
        
        metadata_file = self.output_dir / 'demo_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Demo dataset metadata created: {metadata_file}")
        print(f"📁 Place test images in: {demo_dir}")
        print(f"🐒 ImageNet includes {len(wildlife_classes)} wildlife classes")
        print(f"   Including species found in Colombia:")
        print(f"   • Howler monkey (mono aullador)")
        print(f"   • Spider monkey (mono araña)")
        
        return metadata
    
    def _generate_download_instructions(self) -> str:
        """Generate detailed download instructions for Colombian wildlife data."""
        instructions = """# Download Instructions - Colombian Wildlife Data

## Option 1: iNaturalist (Recommended)

iNaturalist has excellent coverage of Colombian biodiversity with verified observations.

### Step-by-step:

1. **Visit iNaturalist**:
   - URL: https://www.inaturalist.org/observations
   
2. **Set filters**:
   - Place: "Colombia" (or specific department like "Antioquia", "Cundinamarca")
   - Quality grade: "Research Grade" (verified by community)
   - Captive/Cultivated: "No" (wild observations only)
   - Photos: "Yes"

3. **Search by species** (in Spanish or scientific name):
   - Jaguar / Panthera onca
   - Oso de anteojos / Tremarctos ornatus
   - Danta de montaña / Tapirus pinchaque
   - Puma / Puma concolor
   - Chigüiro / Hydrochoerus hydrochaeris
   - Águila arpía / Harpia harpyja
   
4. **Download images**:
   - Click on each observation
   - Right-click image → "Save image as"
   - Save to: `data/wildlife/colombia/{species_name}/`
   
5. **Recommended**: Download 50-100 images per species

### Example URLs:

**Jaguar in Colombia**:
https://www.inaturalist.org/observations?place_id=7562&taxon_name=Panthera%20onca&quality_grade=research

**Spectacled Bear (Oso de Anteojos)**:
https://www.inaturalist.org/observations?place_id=7562&taxon_name=Tremarctos%20ornatus&quality_grade=research

**Capybara (Chigüiro)**:
https://www.inaturalist.org/observations?place_id=7562&taxon_name=Hydrochoerus%20hydrochaeris&quality_grade=research

---

## Option 2: Camera Trap Images (If you have access)

If you have access to camera trap images from Colombian protected areas:

1. Contact:
   - Instituto Humboldt: https://www.humboldt.org.co/
   - Parques Nacionales: https://www.parquesnacionales.gov.co/
   - Local conservation organizations

2. Request sample datasets for research/education

3. Place images in: `data/wildlife/colombia/camera_traps/`

---

## Option 3: Public Datasets with Colombian Species

### Biodiversity Heritage Library (BHL)
- URL: https://www.biodiversitylibrary.org/
- Search: Species + "Colombia"

### GBIF (Global Biodiversity Information Facility)
- URL: https://www.gbif.org/
- Filter by: Colombia + Has images

---

## For Quick Demo (No Downloads)

Use the ImageNet classes we already support:
- Howler monkey (mono aullador) - Class n02492660
- Spider monkey (mono araña) - Class n02494079
- Other wildlife classes (lions, tigers, etc. for comparison)

Just use images from `examples/test_images/` to test the system.

---

## Data Structure

```
data/wildlife/
├── colombia/
│   ├── jaguar/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── spectacled_bear/
│   ├── mountain_tapir/
│   └── ...
├── serengeti/
│   └── (baseline comparison)
└── demo/
    └── (test images)
```

---

## Citation

If you use iNaturalist data, please cite:
- iNaturalist. Available from https://www.inaturalist.org. Accessed [Date].

If you publish results:
- Mention the observers who contributed the observations
- Include observation IDs for reproducibility

---

## Conservation Context

**Why Colombian Wildlife?**

Colombia is one of the world's megadiverse countries:
- 🏆 #1 in bird species (1,954 species)
- 🏆 #1 in orchid species (4,270 species)
- 🏆 #2 in amphibian species (803 species)
- 🏆 #3 in reptile species (537 species)
- 🏆 #4 in mammal species (528 species)

Many species are endemic and endangered, making affordable AI monitoring critical.

**Protected Areas**:
- Parque Nacional Natural Los Katíos (UNESCO World Heritage)
- Serranía de Chiribiquete (largest tropical rainforest national park)
- Sierra Nevada de Santa Marta (highest coastal mountain range)

Affordable wildlife monitoring (RX 580 @ $750) vs expensive solutions ($15,000+) 
enables more conservation projects in Colombia's 59 national parks.

---

**Need help?** Check the main documentation or open an issue on GitHub.
"""
        return instructions
    
    def print_summary(self):
        """Print summary of available datasets."""
        print("\n" + "="*70)
        print("📊 WILDLIFE DATASETS SUMMARY")
        print("="*70)
        
        print("\n🇨🇴 **COLOMBIA WILDLIFE** (Recommended for local impact)")
        print("   Source: iNaturalist")
        print("   Species: 10+ iconic species (jaguar, spectacled bear, etc.)")
        print("   Cost: Free (Creative Commons)")
        print("   Setup time: 30-60 minutes")
        print("   Impact: Demonstrate AI for Colombian conservation")
        
        print("\n🦁 **SNAPSHOT SERENGETI** (Baseline comparison)")
        print("   Source: LILA (Microsoft AI for Earth)")
        print("   Images: 2.65M camera trap images")
        print("   Species: 48 African mammals")
        print("   Cost: Free (public domain)")
        print("   Setup time: 2-3 hours (large download)")
        print("   Impact: Industry-standard benchmark")
        
        print("\n🎯 **DEMO DATASET** (Quick start)")
        print("   Source: ImageNet classes")
        print("   Images: Use your own or examples")
        print("   Species: 40+ wildlife classes")
        print("   Cost: Free")
        print("   Setup time: 5 minutes")
        print("   Impact: Immediate testing")
        
        print("\n💡 **RECOMMENDATION**:")
        print("   Start with DEMO to test the system")
        print("   Then download Colombia data for real-world demo")
        print("   Optionally add Serengeti for comparison")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Download wildlife datasets for RX 580 AI demo'
    )
    
    parser.add_argument(
        '--region',
        choices=['colombia', 'serengeti', 'demo', 'all'],
        default='demo',
        help='Dataset region to prepare'
    )
    
    parser.add_argument(
        '--species',
        type=str,
        default='all',
        help='Species to download (comma-separated or "all")'
    )
    
    parser.add_argument(
        '--num-images',
        type=int,
        default=500,
        help='Target number of images per species'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/wildlife',
        help='Output directory for datasets'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary of available datasets'
    )
    
    args = parser.parse_args()
    
    downloader = WildlifeDatasetDownloader(args.output_dir)
    
    if args.summary:
        downloader.print_summary()
        return
    
    species_list = args.species.split(',') if args.species != 'all' else ['all']
    
    if args.region == 'colombia':
        downloader.download_inaturalist_colombia(species_list, args.num_images)
    elif args.region == 'serengeti':
        downloader.download_snapshot_serengeti_sample(args.num_images)
    elif args.region == 'demo':
        downloader.create_demo_dataset()
    elif args.region == 'all':
        downloader.create_demo_dataset()
        downloader.download_inaturalist_colombia(['all'], args.num_images)
        downloader.download_snapshot_serengeti_sample(args.num_images)
    
    print("\n✅ Dataset preparation complete!")
    print(f"📁 Data directory: {downloader.output_dir}")
    print("\n🚀 Next steps:")
    print("   1. Download actual images following the instructions")
    print("   2. Run the wildlife monitoring demo:")
    print("      python examples/use_cases/wildlife_monitoring.py")
    print()


if __name__ == '__main__':
    main()
