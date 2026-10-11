"""Writes tests/data/bg_docs_small.jsonl.txt: 25 synthetic encyclopedia-like docs (fictional subjects).

Rows follow the LG/AG hf_stream format {"id", "text", "meta"}; paragraphs are separated by a blank
line and section headings sit on their own line, like the wikimedia/wikipedia text field.
Deterministic: python tools/make_bg_docs_small.py tests/data/bg_docs_small.jsonl.txt
"""
from __future__ import annotations

import json
import random
import sys

KINDS = {
    "town": {
        "noun": "town", "lead": [
            "{name} is a small {adj} town in the {region}, on the {side} bank of the {river}.",
            "It had a population of {pop} at the census of {y4}, and serves as the market centre for the surrounding farms.",
            "The town is known for its covered market hall, its stone bridge and a long tradition of {craft}.",
        ],
        "sections": {
            "History": [
                "The first written record of {name} dates from {y1}, when it appears in a land survey as a cluster of farms near a ford.",
                "A charter granted in {y2} allowed the inhabitants to hold a weekly market and to collect tolls on the bridge.",
                "Much of the old centre was destroyed by a fire in {y3}, and the present street plan was laid out during the rebuilding.",
                "During the nineteenth century the town grew slowly, and its population roughly doubled between {y3} and {y4}.",
                "The arrival of the railway in {y3} brought new workshops, a brickworks and a small hotel near the station.",
                "Local records describe a long dispute with the neighbouring village of {other} over grazing rights on the common.",
            ],
            "Geography": [
                "The town lies at an elevation of about {elev} metres, where the valley of the {river} narrows between two low ridges.",
                "The climate is temperate, with mild summers and frequent fog in late autumn.",
                "To the north the land rises towards a plateau of heath and scattered woodland, which is now a protected area.",
                "Several small streams join the {river} within the town boundary, and two of them were culverted in the twentieth century.",
            ],
            "Economy": [
                "Agriculture remains important, and the surrounding farms produce grain, potatoes and dairy products.",
                "The former brickworks closed in {y4}, and its site was later turned into a small business park.",
                "Tourism has grown since the restoration of the market hall, which now hosts a craft fair every summer.",
                "About {pct} percent of the working population commutes to larger towns in the {region}.",
            ],
            "Culture": [
                "The parish church contains a carved wooden screen that local historians date to the fifteenth century.",
                "A town museum, opened in {y4}, displays tools, photographs and records from the brickworks and the old market.",
                "The annual autumn fair, first held in {y2}, still takes place on the meadow beside the river.",
                "A choir founded in {y3} performs in the church and in neighbouring villages throughout the year.",
            ],
        },
    },
    "river": {
        "noun": "river", "lead": [
            "The {name} is a river in the {region}, about {length} kilometres long.",
            "It rises on the slopes of the {other} hills and flows generally {dir} before joining the {river}.",
            "The river gives its name to the {name} valley, a largely agricultural area with several old mill sites.",
        ],
        "sections": {
            "Course": [
                "From its source at about {elev} metres, the river descends quickly through a narrow wooded gorge.",
                "Below the gorge the valley widens, and the river meanders across a broad floodplain of meadows and willow thickets.",
                "Its main tributaries are the {other} Brook and the Linden Water, which join it from the left bank.",
                "In its lower course the river was straightened in {y3} to reduce flooding of the adjacent farmland.",
            ],
            "History": [
                "Mills were recorded along the river as early as {y1}, and by {y2} there were more than a dozen working sites.",
                "A canal connecting the {name} with the {river} was proposed in {y3} but was never completed.",
                "A severe flood in {y4} damaged several bridges and led to the construction of a series of retention basins.",
                "The old towpath along the lower river was converted into a walking and cycling route in {y4}.",
            ],
            "Ecology": [
                "The river supports populations of brown trout, grayling and several species of freshwater mussel.",
                "Otters returned to the upper valley in the early twenty-first century after an absence of several decades.",
                "Water quality improved markedly after a new treatment plant was opened near the mouth of the river in {y4}.",
                "Parts of the floodplain are managed as wet meadows, which are mown late in the summer to protect nesting birds.",
            ],
        },
    },
    "beetle": {
        "noun": "species", "lead": [
            "The {name} is a species of ground beetle found in the {region}.",
            "It was first described in {y2} from specimens collected on dry heathland, and it is now known from a few dozen sites.",
            "Adults are about {size} millimetres long, with a dark bronze body and faint ridges along the wing cases.",
        ],
        "sections": {
            "Description": [
                "The head is broad, with long slender antennae that reach roughly half the length of the body.",
                "The legs are reddish brown, and the front pair is adapted for digging in loose sandy soil.",
                "Males can be distinguished from females by slightly widened segments on the front feet.",
                "The larvae are pale and elongated, with a hardened head capsule and strong jaws.",
            ],
            "Distribution and habitat": [
                "The species prefers open, sunny ground with sparse vegetation and a loose sandy surface.",
                "Most records come from lowland heath, old gravel pits and the edges of pine plantations.",
                "Surveys between {y3} and {y4} found that several historic sites no longer supported the beetle.",
                "It is rarely found above {elev} metres, and it appears to avoid areas with dense grass cover.",
            ],
            "Behaviour": [
                "Adults are active mainly on warm days in late spring, when they hunt small insects on bare ground.",
                "The larvae live in short burrows and ambush prey that passes close to the entrance.",
                "The life cycle usually takes two years, and the larvae overwinter deep in the soil.",
                "Like many ground beetles, the species can release a strong-smelling defensive secretion when disturbed.",
            ],
            "Conservation": [
                "The beetle is listed as vulnerable in the regional red list published in {y4}.",
                "Management of its habitat includes removing invading trees and scraping small patches of bare sand.",
                "A monitoring programme has counted adults at fixed transects every spring since {y4}.",
            ],
        },
    },
    "loom": {
        "noun": "machine", "lead": [
            "The {name} loom is a type of mechanical loom developed in the {region} in the {cent} century.",
            "It was designed by the weaver {founder}, who patented an improved shuttle mechanism in {y2}.",
            "The loom was widely used in small workshops until it was replaced by power looms in the late {cent2} century.",
        ],
        "sections": {
            "Design": [
                "The frame was built of oak and could be assembled by a carpenter without special tools.",
                "Its main innovation was a spring-driven shuttle that allowed a single weaver to produce wider cloth.",
                "A pattern drum with wooden pegs controlled the lifting of the warp threads for simple repeating designs.",
                "Later versions added an iron ratchet that advanced the cloth beam automatically after each pick.",
            ],
            "History": [
                "The first workshop to use the loom opened in {y2}, and within twenty years several hundred were in use.",
                "Guilds in the {region} initially opposed the loom because it reduced the number of workers needed.",
                "By {y3} the design had been copied in neighbouring regions, often with minor changes to the shuttle.",
                "The last commercial workshop using the loom closed in {y4}.",
            ],
            "Legacy": [
                "Several examples survive in museums, and one working loom is demonstrated at a textile museum each weekend.",
                "Historians regard the loom as an important step between hand weaving and the factory system.",
                "A reconstruction built in {y4} was used to study the cloth produced by early workshops.",
            ],
        },
    },
    "festival": {
        "noun": "festival", "lead": [
            "The {name} Festival is an annual festival held in the town of {other} in the {region}.",
            "It takes place over three days in early {month} and attracts around {visitors} visitors.",
            "The festival combines a lantern procession, a market and performances of traditional music.",
        ],
        "sections": {
            "Origins": [
                "Local tradition links the festival to a harvest celebration first recorded in {y1}.",
                "The modern form of the festival dates from {y3}, when a committee of residents revived the procession.",
                "Early accounts describe paper lanterns carried from the church to the river at dusk.",
            ],
            "Programme": [
                "The first evening opens with a procession of lanterns made by schools and local associations.",
                "On the second day the market square hosts stalls selling crafts, baked goods and regional cheeses.",
                "Folk ensembles from the {region} perform on an open-air stage throughout the weekend.",
                "The festival closes with a concert on the riverbank, followed by the release of floating lanterns.",
            ],
            "Organisation": [
                "The festival is run by a volunteer association founded in {y3}, with support from the town council.",
                "Since {y4} the organisers have replaced candles with battery lights to reduce the risk of fire.",
                "Entry to most events is free, and the costs are covered by stall fees and local sponsors.",
            ],
        },
    },
    "observatory": {
        "noun": "observatory", "lead": [
            "The {name} Observatory is an astronomical observatory on a hill above the town of {other} in the {region}.",
            "It was founded in {y2} by a society of amateur astronomers and later passed to the regional university.",
            "Today it is used mainly for teaching, public viewing evenings and the monitoring of variable stars.",
        ],
        "sections": {
            "History": [
                "The society raised funds for the first dome through subscriptions and a public lecture series.",
                "A refracting telescope with a lens of {size} centimetres was installed in {y3}.",
                "During the {cent2} century the observatory contributed timing observations to a national survey.",
                "The university took over the site in {y4} and built a small lecture hall next to the dome.",
            ],
            "Instruments": [
                "The main instrument is a reflecting telescope with a mirror about half a metre across.",
                "The original refractor has been restored and is still used for public viewing of the Moon and planets.",
                "A small radio antenna was added in {y4} for student projects on the hydrogen line.",
            ],
            "Public outreach": [
                "Public viewing evenings are held every Friday when the sky is clear.",
                "The observatory runs summer courses for school pupils, which have been offered since {y4}.",
                "Light pollution from the growing town has reduced the quality of the site for faint objects.",
            ],
        },
    },
    "railway": {
        "noun": "railway line", "lead": [
            "The {name} Railway is a former branch line in the {region}, about {length} kilometres long.",
            "It opened in {y3} to connect the quarries near {other} with the main line, and later carried passengers.",
            "The line closed to passengers in {y4}, and part of the route is now a heritage railway.",
        ],
        "sections": {
            "Construction": [
                "Construction began in {y3}, after a long debate about whether the line should follow the valley or the ridge.",
                "The engineers chose the valley route, which required two viaducts and a short tunnel.",
                "Most of the labour was provided by workers from the quarries, who were paid partly in shares.",
            ],
            "Operation": [
                "At its peak the line carried around {num} tonnes of stone each year.",
                "Passenger services consisted of four trains a day in each direction, with extra trains on market days.",
                "Steam locomotives were replaced by diesel railcars in the middle of the twentieth century.",
                "Freight traffic declined after the largest quarry closed, and the line ran at a loss for many years.",
            ],
            "Preservation": [
                "A preservation society was formed shortly after the closure and leased the first section of the route.",
                "The heritage railway runs steam trains on weekends from spring to autumn.",
                "Volunteers restored one of the viaducts with funding from a regional heritage programme.",
            ],
        },
    },
    "mineral": {
        "noun": "mineral", "lead": [
            "{name}ite is a rare phosphate mineral first described in {y3} from a quarry in the {region}.",
            "It forms small pale green crystals, usually less than {size} millimetres long, in cavities of weathered granite.",
            "The mineral is named after the hill of {name}, where the first specimens were found.",
        ],
        "sections": {
            "Properties": [
                "The crystals are prismatic and have a glassy lustre and a white streak.",
                "It has a hardness of about four on the Mohs scale and breaks with an uneven fracture.",
                "Under ultraviolet light some specimens show a weak yellow fluorescence.",
            ],
            "Occurrence": [
                "It occurs together with quartz, feldspar and other secondary phosphate minerals.",
                "Apart from the type locality, it has been reported from only a few sites in the {region}.",
                "Collectors visited the original quarry frequently until it was flooded in {y4}.",
            ],
            "Research": [
                "Its crystal structure was determined in {y4} using single-crystal diffraction.",
                "Researchers have used the mineral to study how phosphate is released during the weathering of granite.",
                "Specimens from the type locality are kept in the collection of the regional natural history museum.",
            ],
        },
    },
    "dish": {
        "noun": "dish", "lead": [
            "{name} is a traditional dish from the {region}, made of {ingredient}, onions and root vegetables.",
            "It is usually cooked slowly in a clay pot and served in winter with dark bread.",
            "Recipes vary between villages, and several towns hold an annual cooking contest for the dish.",
        ],
        "sections": {
            "History": [
                "The earliest known recipe appears in a household book from {y2}.",
                "The dish was originally prepared on baking days, when the bread oven was still warm.",
                "In the {cent2} century it became a common meal in the inns along the old trade road.",
            ],
            "Preparation": [
                "The vegetables are cut into large pieces and layered with the {ingredient} in the pot.",
                "A small amount of beer or stock is added, and the pot is sealed with a lid of dough.",
                "The dish is cooked for three to four hours at a low temperature.",
                "Some cooks add dried plums or juniper berries to give the sauce a sweeter flavour.",
            ],
            "Cultural significance": [
                "The dish is associated with the first snowfall, when families traditionally prepare it together.",
                "A local association has campaigned since {y4} for the dish to receive a protected regional designation.",
                "Cookbooks published in the {region} usually include at least one version of the recipe.",
            ],
        },
    },
    "library": {
        "noun": "library", "lead": [
            "The {name} Library is a public library in the town of {other} in the {region}.",
            "It was established in {y2} with a donation of books from a local merchant and moved to its present building in {y3}.",
            "The library holds about {num} volumes, including a collection of local newspapers and maps.",
        ],
        "sections": {
            "History": [
                "In its first decades the library was open only two evenings a week and was run by volunteers.",
                "The present building was designed by the architect {founder} and opened in {y3}.",
                "A children's reading room was added in {y4}, followed by a computer room a few years later.",
            ],
            "Collections": [
                "The local history collection contains parish records, photographs and the archive of the town newspaper.",
                "A set of hand-drawn maps from the {cent} century shows the field boundaries around the town.",
                "The library also holds the papers of several local writers and teachers.",
            ],
            "Services": [
                "Besides lending books, the library offers language courses, reading groups and help with digital services.",
                "A mobile library visits the surrounding villages every second week.",
                "Since {y4} the library has digitised part of its newspaper collection for online access.",
            ],
        },
    },
}

GENERIC = {
    "Recent developments": [
        "In {y4} the regional government commissioned a survey of the {noun} and its surroundings.",
        "The survey recommended better signposting, a small visitor centre and regular maintenance work.",
        "Since then a local association has organised guided walks and talks on the history of the area.",
        "A booklet describing the {noun} was published by the regional historical society in {y4}.",
        "Students from the university in the {region} have carried out several field projects on the site.",
        "Funding from a heritage programme paid for repairs and for new information boards in two languages.",
        "Visitor numbers have increased steadily, especially during the summer holidays.",
        "Plans for a new access road were discussed in {y4} but were postponed after objections from residents.",
    ],
    "Names and records": [
        "The name {name} is first recorded in a document from {y1}, where it is spelled in several different ways.",
        "Historians have suggested that the name comes from an old word for a clearing or a bend in a stream.",
        "Older maps of the {region} show the {noun} under a slightly different name until the late {cent2} century.",
        "Parish registers and tax lists from {y2} provide some of the earliest details about the people of the area.",
        "A collection of letters from {y3} describes daily life in the district and mentions the {noun} several times.",
        "Most of the surviving records are kept in the regional archive, where they have been catalogued since {y4}.",
    ],
}

NAMES = ["Varnholt", "Ellsmere", "Corvane", "Tallis", "Osk", "Brenmoor", "Quillan", "Hesketh", "Arden", "Morrow",
         "Thistlewick", "Calder", "Dunmere", "Fennick", "Galloway", "Harrowgate", "Ivel", "Kestrel", "Lowther",
         "Marisdale", "Norwick", "Pellam", "Rookfield", "Selwyn", "Wexcombe"]
OTHERS = ["Ashby", "Brindle", "Coldwater", "Dray", "Eskdale", "Foxley", "Greystone", "Holm", "Kirkby", "Larch"]
REGIONS = ["northern uplands", "eastern lowlands", "lake district of Vessmark", "coastal plain of Orland",
           "hill country of Bray", "central valley of Amsel"]
RIVERS = ["River Amsel", "River Dorr", "River Velt", "River Ise"]
FOUNDERS = ["Anna Kessler", "Tomas Brandt", "Elise Varga", "Johan Merle", "Clara Weiss", "Pieter Holm"]
ORDER = ["town", "river", "beetle", "loom", "festival", "observatory", "railway", "mineral", "dish", "library"]


def make_doc(i: int, rng: random.Random) -> tuple[str, str]:
    kind = ORDER[i % len(ORDER)]
    spec = KINDS[kind]
    y1 = rng.randint(1150, 1450)
    y2 = rng.randint(y1 + 40, 1720)
    y3 = rng.randint(max(y2 + 20, 1780), 1905)
    y4 = rng.randint(1950, 2019)
    slots = {
        "name": NAMES[i % len(NAMES)], "other": rng.choice(OTHERS), "region": rng.choice(REGIONS),
        "river": rng.choice(RIVERS), "founder": rng.choice(FOUNDERS), "side": rng.choice(["left", "right"]),
        "adj": rng.choice(["quiet", "historic", "rural", "riverside"]), "dir": rng.choice(["north", "east", "south-west"]),
        "craft": rng.choice(["basket weaving", "pottery", "clock making", "wood carving"]),
        "ingredient": rng.choice(["pork", "beef", "lamb", "smoked sausage"]),
        "month": rng.choice(["September", "October", "November"]),
        "y1": y1, "y2": y2, "y3": y3, "y4": y4, "pop": f"{rng.randint(2, 19)},{rng.randint(100, 999)}",
        "elev": rng.randint(80, 900), "length": rng.randint(18, 140), "size": rng.randint(3, 25),
        "num": f"{rng.randint(20, 480)},000", "pct": rng.randint(15, 60), "visitors": f"{rng.randint(5, 60)},000",
        "cent": rng.choice(["seventeenth", "eighteenth"]), "cent2": "nineteenth", "noun": spec["noun"],
    }
    paras = [" ".join(s.format(**slots) for s in spec["lead"])]
    for head, pool in spec["sections"].items():
        paras.append(head)
        drop = rng.randrange(len(pool)) if len(pool) >= 4 else -1      # keep sentence order, vary the content
        sents = [x for j, x in enumerate(pool) if j != drop]
        cut = rng.randint(2, max(2, len(sents) - 1))
        paras.append(" ".join(s.format(**slots) for s in sents[:cut]))
        if len(sents) - cut >= 2:
            paras.append(" ".join(s.format(**slots) for s in sents[cut:]))
    target_words = rng.randint(420, 720)
    for head in rng.sample(sorted(GENERIC), k=2):
        if len(" ".join(paras).split()) >= target_words:
            break
        sents = GENERIC[head][:]
        paras.append(head)
        half = len(sents) // 2
        for chunk in (sents[:half], sents[half:]):
            if len(" ".join(paras).split()) < target_words:
                paras.append(" ".join(s.format(**slots) for s in chunk))
    return slots["name"] if kind != "mineral" else slots["name"] + "ite", "\n\n".join(paras)


def main(out: str) -> None:
    rng = random.Random(20261011)
    rows = []
    for i in range(25):
        title, text = make_doc(i, rng)
        words = len(text.split())
        assert 300 <= words <= 800 and len(text) >= 2000, (i, words, len(text))
        rows.append({"id": f"bg_synth-{i}", "text": text, "meta": {"title": title, "synthetic": True}})
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(out, len(rows), [len(r["text"].split()) for r in rows])


if __name__ == "__main__":
    main(sys.argv[1])
