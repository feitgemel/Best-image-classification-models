import torch 
import os 
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO 

#Define ontology

# List of objects to generate
categories = ['Lion', 'African Elephant', 'Leopard', 'Rhinocerous', 'Cape Buffalo', 'Cheetah' , 'Giraffe', 'Zebra' , 'Hippo' , 'Crocodile' , 'Wildebeest' , 'Warthhog' ]

# Base directory to save the generated images
base_dir = "/mnt/d/Data-sets/synthetic/Animals-Object-Detection/images"

ontology = CaptionOntology({
    "Lion": "Lion",
    "African Elephant": "African Elephant",
    "Leopard": "Leopard",
    "Rhinocerous": "Rhinocerous",
    "Cape Buffalo": "Cape Buffalo",
    "Cheetah": "Cheetah",
    "Giraffe": "Giraffe",
    "Zebra": "Zebra",
    "Hippo": "Hippo",
    "Crocodile": "Crocodile",
    "Wildebeest": "Wildebeest",
    "Warthhog": "Warthhog"
})

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

DATASET_DIR_PTH = "/mnt/d/Data-sets/synthetic/Animals-Object-Detection/dataset"

base_model = GroundingDINO(ontology=ontology, 
                           box_threshold=BOX_THRESHOLD,
                           text_threshold=TEXT_THRESHOLD)


dataset = base_model.label(input_folder=base_dir, extension=".png", output_folder=DATASET_DIR_PTH)


