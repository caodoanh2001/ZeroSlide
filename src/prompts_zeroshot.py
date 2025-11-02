def brca_prompts():
    prompts = [
        # Labels
        [
            'invasive ductal carcinoma',
            'breast invasive ductal carcinoma',
            'invasive ductal carcinoma of the breast',
            'invasive carcinoma of the breast, ductal pattern',
            'idc',
            # 'invasive ductal carcinoma, which is the most common type of invasive breast cancer. It begins in the lining of the milk ducts (thin tubes that carry milk from the lobules of the breast to the nipple) and spreads outside the ducts to surrounding normal tissue. Invasive ductal carcinoma can also spread through the blood and lymph systems to other parts of the body. Also called infiltrating ductal carcinoma.'
            ],
        [
            'invasive lobular carcinoma',
            'breast invasive lobular carcinoma',
            'invasive lobular carcinoma of the breast',
            'invasive carcinoma of the breast, lobular pattern',
            'ilc',
            # 'invasive lobular carcinoma, which is a type of invasive breast cancer that begins in the lobules (milk glands) of the breast and spreads to surrounding normal tissue. It can also spread through the blood and lymph systems to other parts of the body. Also called infiltrating lobular carcinoma.'
            ]
    ]

    templates = [
                "CLASSNAME.",
                "a photomicrograph showing CLASSNAME.",
                "a photomicrograph of CLASSNAME.",
                "an image of CLASSNAME.",
                "an image showing CLASSNAME.",
                "an example of CLASSNAME.",
                "CLASSNAME is shown.",
                "this is CLASSNAME.",
                "there is CLASSNAME.",
                "a histopathological image showing CLASSNAME.",
                "a histopathological image of CLASSNAME.",
                "a histopathological photograph of CLASSNAME.",
                "a histopathological photograph showing CLASSNAME.",
                "shows CLASSNAME.",
                "presence of CLASSNAME.",
                "CLASSNAME is present.",
                "an H&E stained image of CLASSNAME.",
                "an H&E stained image showing CLASSNAME.",
                "an H&E image showing CLASSNAME.",
                "an H&E image of CLASSNAME.",
                "CLASSNAME, H&E stain.",
                "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return prompts, templates

def nsclc_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
            # https://www.lungevity.org/
            # 'lung adenocarcinoma, which is categorized as such by how the cancer cells look under a microscope. Lung adenocarcinoma starts in glandular cells, which secrete substances such as mucus, and tends to develop in smaller airways, such as alveoli. Lung adenocarcinoma is usually located more along the outer edges of the lungs. Lung adenocarcinoma tends to grow more slowly than other lung cancers.'
            ],
        [
            'squamous cell carcinoma',
            'lung squamous cell carcinoma',
            'squamous cell carcinoma of the lung',
            'lusc',
            # 'squamous cell lung cancer, or squamous cell carcinoma of the lung, which is one type of non-small cell lung cancer (NSCLC). Squamous cell lung cancer is categorized as such by how the cells look under a microscope. Squamous cell lung cancer begins in the squamous cells—thin, flat cells that look like fish scales when seen under a microscope. They line the inside of the airways in the lungs. Squamous cell lung cancer is also called epidermoid carcinoma.'
            ],
    ]

    templates = [
                "CLASSNAME.",
                "a photomicrograph showing CLASSNAME.",
                "a photomicrograph of CLASSNAME.",
                "an image of CLASSNAME.",
                "an image showing CLASSNAME.",
                "an example of CLASSNAME.",
                "CLASSNAME is shown.",
                "this is CLASSNAME.",
                "there is CLASSNAME.",
                "a histopathological image showing CLASSNAME.",
                "a histopathological image of CLASSNAME.",
                "a histopathological photograph of CLASSNAME.",
                "a histopathological photograph showing CLASSNAME.",
                "shows CLASSNAME.",
                "presence of CLASSNAME.",
                "CLASSNAME is present.",
                "an H&E stained image of CLASSNAME.",
                "an H&E stained image showing CLASSNAME.",
                "an H&E image showing CLASSNAME.",
                "an H&E image of CLASSNAME.",
                "CLASSNAME, H&E stain.",
                "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return prompts, templates


def rcc_prompts():
    prompts = [
        [
            'clear cell renal cell carcinoma',
            'renal cell carcinoma, clear cell type',
            'renal cell carcinoma of the clear cell type',
            'clear cell rcc',
            # 'clear cell renal cell carcinoma, which is a type of kidney cancer in which the cells look clear or very pale when viewed under a microscope. Clear cell renal cell carcinoma begins in cells that line tiny tubes in the kidney. These tubes return filtered nutrients, fluids, and other substances that the body needs back to the blood. Clear cell renal cell carcinoma is the most common type of kidney cancer in adults. People with an inherited condition called von Hippel-Lindau syndrome are at an increased risk of developing clear cell renal cell carcinoma. Also called ccRCC, clear cell renal cell cancer, and conventional renal cell carcinoma.'
        ],
        [
            'papillary renal cell carcinoma',
            'renal cell carcinoma, papillary type',
            'renal cell carcinoma of the papillary type',
            'papillary rcc',
            # 'papillary renal cell carcinoma, which is a type of kidney cancer that forms in the lining of the tiny tubes in the kidney that return filtered substances that the body needs back to the blood and remove extra fluid and waste as urine. Most papillary tumors look like long, thin finger-like growths under a microscope. There are two types of papillary renal cell carcinoma: type 1 and type 2. Type 1 tends to grow slowly and spread to other parts of the body less often than type 2. Patients with a genetic disorder called hereditary papillary renal cancer have an increased risk of type 1 papillary renal cell carcinoma. Patients with a genetic disorder called hereditary leiomyomatosis and renal cell cancer have an increased risk of type 2 papillary renal cell carcinoma. Also called papillary kidney cancer and PRCC.'
        ],
        [
            'chromophobe renal cell carcinoma',
            'renal cell carcinoma, chromophobe type',
            'renal cell carcinoma of the chromophobe type',
            'chromophobe rcc',
            'chromophobe renal cell carcinoma, which is a rare type of kidney cancer that forms in the cells lining the small tubules in the kidney. These small tubules help filter waste from the blood, making urine.'
        ]
            ]

    templates = [
                "CLASSNAME.",
                "a photomicrograph showing CLASSNAME.",
                "a photomicrograph of CLASSNAME.",
                "an image of CLASSNAME.",
                "an image showing CLASSNAME.",
                "an example of CLASSNAME.",
                "CLASSNAME is shown.",
                "this is CLASSNAME.",
                "there is CLASSNAME.",
                "a histopathological image showing CLASSNAME.",
                "a histopathological image of CLASSNAME.",
                "a histopathological photograph of CLASSNAME.",
                "a histopathological photograph showing CLASSNAME.",
                "shows CLASSNAME.",
                "presence of CLASSNAME.",
                "CLASSNAME is present.",
                "an H&E stained image of CLASSNAME.",
                "an H&E stained image showing CLASSNAME.",
                "an H&E image showing CLASSNAME.",
                "an H&E image of CLASSNAME.",
                "CLASSNAME, H&E stain.",
                "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return prompts, templates

def esca_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'esophageal adenocarcinoma',
            'adenocarcinoma of the esophagus',
            'esad',
            # https://www.mayoclinic.org/diseases-conditions/esophageal-cancer/symptoms-causes/syc-20356084
            # 'esophageal adenocarcinoma, which begins in the cells of the glands in the esophagus. These glands produce mucus. Adenocarcinoma happens most often in the lower part of the esophagus. Adenocarcinoma is the most common form of esophageal cancer in the United States. It affects mostly white men.'
            ],
        [
            'squamous cell carcinoma',
            'esophageal squamous cell carcinoma',
            'squamous cell carcinoma of the esophagus',
            'essc',
            # https://www.mayoclinic.org/diseases-conditions/esophageal-cancer/symptoms-causes/syc-20356084
            # 'esophageal adenocarcinoma which begins in the flat, thin cells that line the surface of the esophagus. Squamous cell carcinoma happens most often in the upper and middle parts of the esophagus. Squamous cell carcinoma is the most common esophageal cancer worldwide.'
        ]
            ]

    templates = [
                "CLASSNAME.",
                "a photomicrograph showing CLASSNAME.",
                "a photomicrograph of CLASSNAME.",
                "an image of CLASSNAME.",
                "an image showing CLASSNAME.",
                "an example of CLASSNAME.",
                "CLASSNAME is shown.",
                "this is CLASSNAME.",
                "there is CLASSNAME.",
                "a histopathological image showing CLASSNAME.",
                "a histopathological image of CLASSNAME.",
                "a histopathological photograph of CLASSNAME.",
                "a histopathological photograph showing CLASSNAME.",
                "shows CLASSNAME.",
                "presence of CLASSNAME.",
                "CLASSNAME is present.",
                "an H&E stained image of CLASSNAME.",
                "an H&E stained image showing CLASSNAME.",
                "an H&E image showing CLASSNAME.",
                "an H&E image of CLASSNAME.",
                "CLASSNAME, H&E stain.",
                "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return prompts, templates

def tgct_prompts():
    prompts = [
        # Labels
        [
            'seminoma',
            'testicular seminoma',
            'seminoma of the testis',
            # https://www.ncbi.nlm.nih.gov/books/NBK448137/
            # 'testicular seminoma. There are three main pathologic categories of testicular seminoma: classical, spermatocytic, and seminoma with syncytiocytotrophoblastic cells. Spermatocytic type is rare, occurs in older men, and appears to have a better prognosis. The syncytiocytotrophoblastic subtype is associated with increased serum βhCG levels. A seminoma with a high mitotic index (>3 mitotic figures/HPF) is designated an anaplastic seminoma. The name implies a more aggressive tumor, but research fails to support that concern.  '
            ],
        [
            'mixed germ cell tumor',
            'testicular mixed germ cell tumor',
            'mixed germ cell tumor of the testis',
            # https://www.cancer.gov/publications/dictionaries/cancer-terms/def/mixed-germ-cell-tumor,
            # 'mixed germ cell tumor of the testis, which is a rare type of cancer that is made up of at least two different types of germ cell tumors (tumors that begin in cells that form sperm or eggs). These may include choriocarcinoma, embryonal carcinoma, yolk sac tumor, teratoma, and seminoma. Mixed germ cell tumors occur most often in the ovary or testicle'
        ],
            ]

    templates = [
                "CLASSNAME.",
                "a photomicrograph showing CLASSNAME.",
                "a photomicrograph of CLASSNAME.",
                "an image of CLASSNAME.",
                "an image showing CLASSNAME.",
                "an example of CLASSNAME.",
                "CLASSNAME is shown.",
                "this is CLASSNAME.",
                "there is CLASSNAME.",
                "a histopathological image showing CLASSNAME.",
                "a histopathological image of CLASSNAME.",
                "a histopathological photograph of CLASSNAME.",
                "a histopathological photograph showing CLASSNAME.",
                "shows CLASSNAME.",
                "presence of CLASSNAME.",
                "CLASSNAME is present.",
                "an H&E stained image of CLASSNAME.",
                "an H&E stained image showing CLASSNAME.",
                "an H&E image showing CLASSNAME.",
                "an H&E image of CLASSNAME.",
                "CLASSNAME, H&E stain.",
                "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return prompts, templates

def cesc_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'cervical adenocarcinoma',
            'adenocarcinoma of the cervix uteri',
            # https://www.cancerresearchuk.org/about-cancer/cervical-cancer/stages-types-grades/types-and-grades#:~:text=squamous%20cell%20cancers.-,Adenocarcinoma,more%20common%20in%20recent%20years.
            # 'cervical adenocarcinoma, which is a cancer that starts in the gland cells that produce mucus. The cervix has glandular cells scattered along the inside of the passage that runs from the cervix to the womb (endocervix).'
            ],
        [
            'squamous cell carcinoma',
            'cervical squamous cell carcinoma',
            'squamous cell carcinoma of the cervix uteri',
            # https://www.ncbi.nlm.nih.gov/books/NBK559075/
            # 'cervical adenocarcinoma. Similar to squamous cell carcinoma elsewhere characterized by squamous differentiation where it can be affected by grade and degree of differentiation (well-differentiated, moderately differentiated, and poorly differentiated), (ranging from abundant to scant eosinophilic cytoplasm with or without intercellular bridges to small undifferentiated (primitive) cells), hyperchromatic nuclei, high nucleus to cytoplasmic ratio, mitosis (few to high) and abundant to lack of keratinization. WHO histologically classifies epithelial tumors of the cervix into squamous tumors and their precursor (squamous intraepithelial lesion) and squamous cell carcinoma into keratinizing, non-keratinizing, basaloid, warty, papillary, verrucous, squamotransitional, and lymphoepithelioma-like, each of these types has a specific morphologic and immunohistochemical characteristic. Some of the features that can help determine the invasiveness of squamous cell carcinoma include the presence of stromal inflammation, stromal desmoplastic reaction, numerous single or small clusters of highly dysplastic epithelial cells that look different from the ones in the rete ridges, elongated rete ridges, and loss of nuclear polarity. The histological grades include well-differentiated, moderately differentiated, and poorly differentiated. Recent data suggest that recurrence-free survival was significantly reduced in patients with poorly differentiated tumors.'
        ],
            ]

    templates = [
                "CLASSNAME.",
                "a photomicrograph showing CLASSNAME.",
                "a photomicrograph of CLASSNAME.",
                "an image of CLASSNAME.",
                "an image showing CLASSNAME.",
                "an example of CLASSNAME.",
                "CLASSNAME is shown.",
                "this is CLASSNAME.",
                "there is CLASSNAME.",
                "a histopathological image showing CLASSNAME.",
                "a histopathological image of CLASSNAME.",
                "a histopathological photograph of CLASSNAME.",
                "a histopathological photograph showing CLASSNAME.",
                "shows CLASSNAME.",
                "presence of CLASSNAME.",
                "CLASSNAME is present.",
                "an H&E stained image of CLASSNAME.",
                "an H&E stained image showing CLASSNAME.",
                "an H&E image showing CLASSNAME.",
                "an H&E image of CLASSNAME.",
                "CLASSNAME, H&E stain.",
                "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return prompts, templates