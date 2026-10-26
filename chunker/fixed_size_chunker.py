import math
from langchain_core.documents import Document

def predict_chunk_count(text_length: int, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    """Calculates the expected number of chunks using the sliding window formula."""
    if text_length <= 0:
        return 0
    if text_length <= chunk_size:
        return 1
    return math.ceil((text_length - chunk_overlap) / (chunk_size - chunk_overlap))

def fixed_size_chunker(text_content : str , chunk_size = 500 , chunk_overlap=50) :
    start = 0
    i=0
    if chunk_overlap >= chunk_size:
      raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    chunks = []
    while start < len(text_content):
        end = start + chunk_size
        chunk = text_content[start : end]
        chunks.append(Document(page_content=chunk , 
          metadata ={
            "chunk_id" :f"chunk_{i:03d}" , # produces chunk_000 , chunk_001 , etc .
            "chunk_index" : i , 
          }))
        start = end - chunk_overlap
        i+=1
   
    return chunks 

doc = """A forest is an ecosystem characterized by a dense community of trees.[2] Hundreds of definitions of forest are used throughout the world, incorporating factors such as tree density, tree height, land use, legal standing, and ecological function.[3][4][5] The United Nations' Food and Agriculture Organization (FAO) defines a forest as, "Land spanning more than 0.5 hectares with trees higher than 5 meters and a canopy cover of more than 10 percent, or trees able to reach these thresholds in situ. It does not include land that is predominantly under agricultural or urban use."[6] Using this definition, Global Forest Resources Assessment 2025 found that forests covered 4.14 billion hectares (10.2 billion acres; 41.4 million square kilometres; 16.0 million square miles), or approximately 31 percent of the world's land area in 2025.[7]

Forests are the largest terrestrial ecosystems of Earth by area, and are found around the globe.[8] 45 percent of forest land is in the tropical latitudes. The next largest share of forests is found in subarctic climates, followed by temperate, and subtropical zones.[9]

Forests account for 75% of the gross primary production of the Earth's biosphere, and contain 80% of the Earth's plant biomass. Net primary production is estimated at 21.9 gigatonnes of biomass per year for tropical forests, 8.1 for temperate forests, and 2.6 for boreal forests.[8]

Forests form distinctly different biomes at different latitudes and elevations, and with different precipitation and evapotranspiration rates.[10] These biomes include boreal forests in subarctic climates, tropical moist forests and tropical dry forests around the Equator, and temperate forests at the middle latitudes. Forests form in areas of the Earth with high rainfall, while drier conditions produce a transition to savanna. However, in areas with intermediate rainfall levels, forest transitions to savanna rapidly when the percentage of land that is covered by trees drops below 40 to 45 percent.[11] Research conducted in the Amazon rainforest shows that trees can alter rainfall rates across a region, releasing water from their leaves in anticipation of seasonal rains to trigger the wet season early. Because of this, seasonal rainfall in the Amazon begins two to three months earlier than the climate would otherwise allow.[12][13] Deforestation in the Amazon and anthropogenic climate change hold the potential to interfere with this process, causing the forest to pass a threshold where it transitions into savanna.[14]

Deforestation threatens many forest ecosystems. Deforestation occurs when humans remove trees from a forested area by cutting or burning, either to harvest timber or to make way for farming. Most deforestation today occurs in tropical forests. The vast majority of this deforestation is because of the production of four commodities: wood, beef, soy, and palm oil.[15] Over the past 2,000 years, the area of land covered by forest in Europe has been reduced from 80% to 34%. Large areas of forest have also been cleared in China and in the eastern United States,[16] in which only 0.1% of land was left undisturbed.[17] Almost half of Earth's forest area (49 percent) is relatively intact, while 9 percent is found in fragments with little or no connectivity. Tropical rainforests and boreal coniferous forests are the least fragmented, whereas subtropical dry forests and temperate oceanic forests are among the most fragmented. Roughly 80 percent of the world's forest area is found in patches larger than 1 million hectares (2.5 million acres). The remaining 20 percent is located in more than 34 million patches around the world – the vast majority less than 1,000 hectares (2,500 acres) in size.[9]

Human society and forests can affect one another positively or negatively.[18] Forests provide ecosystem services to humans and serve as tourist attractions. Forests can also affect people's health. Human activities, including unsustainable use of forest resources, can negatively affect forest ecosystems.[19]

Definitions

The heavily traversed Adirondack Mountains of Upstate New York form the southernmost part of the Eastern forest-boreal transition ecoregion, constituting part of the world's taiga biome.
Although the word forest is commonly used, there is no universally recognised precise definition, with more than 800 definitions of forest used around the world.[5] Although a forest is usually defined by the presence of trees, under many definitions an area completely lacking trees may still be considered a forest if it grew trees in the past, will grow trees in the future,[20] or was legally designated as a forest regardless of vegetation type.[21][22]

There are three broad categories of definitions of forest in use: administrative, land use, and land cover.[21] Administrative definitions are legal designations, and may not reflect the type of vegetation that grows upon the land; an area can be legally designated "forest" even if no trees grow on it.[21] Land-use definitions are based on the primary purpose the land is used for. Under a land-use definition, any area used primarily for harvesting timber, including areas that have been cleared by harvesting, disease, fire, or for the construction of roads and infrastructure, is still defined as a forest, even if they contain no trees. Land-cover definitions define forests based upon the density of trees, area of tree canopy cover, or area of the land occupied by the cross-section of tree trunks (basal area) meeting a particular threshold.[21] This type of definition depends upon the presence of trees sufficient to meet the threshold, or at least of immature trees that are expected to meet the threshold once they mature.[21]

Under land-cover definitions, there is considerable variation on where the cutoff points are between a forest, woodland, and savanna. Under some definitions, to be considered a forest requires very high levels of tree canopy cover, from 60% to 100%,[23] which excludes woodlands and savannas, which have a lower canopy cover. Other definitions consider savannas to be a type of forest, and include all areas with tree canopies over 10%.[20]

Some areas covered with trees are legally defined as agricultural areas, for example Norway spruce plantations, under Austrian forest law, when the trees are being grown as Christmas trees and are below a certain height.

Etymology

Since the 13th century, the Niepołomice Forest in Poland has had special use and protection. In this view from space, different coloration can indicate different functions.[24]
The word forest derives from the Old French forest (also forès), denoting "forest, vast expanse covered by trees"; forest was first introduced into English as the word denoting wild land set aside for hunting[25] without necessarily having trees on the land.[26] Possibly a borrowing, probably via Frankish or Old High German, of the Medieval Latin foresta, denoting "open wood", Carolingian scribes first used foresta in the capitularies of Charlemagne, specifically to denote the royal hunting grounds of the king. The word was not endemic to the Romance languages, e.g., native words for forest in the Romance languages derived from the Latin silva, which denoted "forest" and "wood(land)" (cf. the English sylva and sylvan; the Italian, Spanish, and Portuguese selva; the Romanian silvă; the Old French selve). Cognates of forest in Romance languages—e.g., the Italian foresta, Spanish and Portuguese floresta, etc.—are all ultimately derivations of the French word.

The precise origin of Medieval Latin foresta is obscure. Some authorities claim the word derives from the Late Latin p"



"""
chunked_docs = fixed_size_chunker(text_content=doc)
predicted_chunks = predict_chunk_count(len(doc), chunk_size=500, chunk_overlap=50)

print(f"Total characters in doc: {len(doc)}")
print(f"Predicted chunks: {predicted_chunks}")
print(f"Actual chunks created: {len(chunked_docs)}")
print(f"First chunk metadata: {chunked_docs[0].metadata}")
print(f"Last chunk metadata: {chunked_docs[-1].metadata}")
