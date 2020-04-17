from typing import Dict, Text, Any, List, Union, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import SlotSet

from joblib import load
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ElicitationForm(FormAction):
    """Example of a custom form action"""

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "elicitation_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["bug", "beverage", "second_person_plural", 
        "cot_caught", "rain_sun", "crawfish", "halloween",
        "sandwich", "side_road", "shoes", "highway", "yard_sale",
        "rubbernecking", "frosting", "lawyer", "kitty_corner",
        "firefly", "verge", "brew_thru", "water_fountain"]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {
            "bug":[self.from_entity(
                entity="bug", 
                intent="inform"),
                self.from_text()],
            "beverage": [self.from_entity(
                entity="beverage", 
                intent="inform"), 
                self.from_text()],
            "second_person_plural": [self.from_entity(
                entity="second_person_plural", 
                intent="inform"),
                self.from_text()],
            "cot_caught": [self.from_entity(
                entity="cot_caught", 
                intent="inform"),
                self.from_text()],
            "rain_sun": [self.from_entity(
                entity="rain_sun", 
                intent="inform"),
                self.from_text()],
            "crawfish": [self.from_entity(
                entity="crawfish", 
                intent="inform"),
                self.from_text()],
            "halloween": [self.from_entity(
                entity="halloween", 
                intent="inform"),
                self.from_text()],
            "sandwich": [self.from_entity(
                entity="sandwich", 
                intent="inform"),
                self.from_text()],
            "side_road": [self.from_entity(
                entity="side_road", 
                intent="inform"),
                self.from_text()],
            "shoes": [self.from_entity(
                entity="shoes", 
                intent="inform"),
                self.from_text()],
            "highway": [self.from_entity(
                entity="highway", 
                intent="inform"),
                self.from_text()],
            "yard_sale": [self.from_entity(
                entity="yard_sale", 
                intent="inform"),
                self.from_text()],
            "rubbernecking": [self.from_entity(
                entity="rubbernecking", 
                intent="inform"),
                self.from_text()],
            "frosting": [self.from_entity(
                entity="frosting", 
                intent="inform"),
                self.from_text()],
            "lawyer": [self.from_entity(
                entity="lawyer", 
                intent="inform"),
                self.from_text()],
            "kitty_corner": [self.from_entity(
                entity="kitty_corner", 
                intent="inform"),
                self.from_text()],
            "firefly": [self.from_entity(
                entity="firefly", 
                intent="inform"),
                self.from_text()],
            "verge": [self.from_entity(
                entity="verge", 
                intent="inform"),
                self.from_text()],
            "brew_thru": [self.from_entity(
                entity="brew_thru", 
                intent="inform"),
                self.from_text()],
            "water_fountain": [self.from_entity(
                entity="water_fountain", 
                intent="inform"),
                self.from_text()]
        }

    # validate user answers
    # TODO: validate 'bug' slot
    @staticmethod
    def answers_db() -> Dict[str, List]:
        """Database of supported cuisines"""

        return {'second_person_plural': ['you all',
            'yous, youse',
            'you lot',
            'you guys',
            "you 'uns",
            'yins',
            'you',
            'other',
            "y'all"],
            'cot_caught': ['different', 'same'],
            'rain_sun': ['sunshower',
            'the wolf is giving birth',
            'the devil is beating his wife',
            "monkey's wedding",
            "fox's wedding",
            'pineapple rain',
            'liquid sun',
            'I have no term or expression for this',
            'other'],
            'crawfish': ['crawfish',
            'crayfish',
            'craw',
            'crowfish',
            'crawdad',
            'mudbug',
            'I have no word for this critter',
            'other'],
            'halloween': ['gate night',
            'trick night',
            'mischief night',
            'cabbage night',
            'goosy night',
            "devil's night",
            "devil's eve",
            'I have no word for this',
            'other'],
            'sandwich': ['sub',
            'grinder',
            'hoagie',
            'hero',
            'poor boy',
            'bomber',
            'Italian sandwich',
            'baguette',
            'sarney',
            'I have no word for this',
            'other'],
            'side_road': ['frontage road',
            'service road',
            'access road',
            'feeder road',
            'gateway',
            'we have them but I have no word for them',
            "I've never heard of this concept",
            'other'],
            'beverage': ['soda',
            'pop',
            'coke',
            'tonic',
            'soft drink',
            'lemonade',
            'cocola',
            'fizzy drink',
            'dope',
            'other'],
            'shoes': ['sneakers',
            'shoes',
            'gymshoes',
            'sand shoes',
            'jumpers',
            'tennis shoes',
            'running shoes',
            'runners',
            'trainers',
            'I have no general word for this',
            'other'],
            'highway': ['highway',
            'freeway',
            'parkway',
            'turnpike',
            'expressway',
            'throughway/thru-way',
            'a freeway is bigger than a highway',
            'a freeway is free',
            'a freeway has limited access',
            'other'],
            'yard_sale': ['tag sale',
            'yard sale',
            'garage sale',
            'rummage sale',
            'thrift sale',
            'stoop sale',
            'carport sale',
            'sidewalk sale',
            'jumble',
            'car boot sale',
            'car boot',
            'patio sale',
            'other'],
            'rubbernecking': ['rubberneck',
            'rubbernecking',
            'rubbernecking is the activity',
            "gapers' block",
            "gapers' delay",
            'Lookie Lou',
            'curiosity delay',
            'gawk block',
            'I have no word for this',
            'other'],
            'frosting': ['frosting',
            'icing',
            'icing is thinner than frosting, white, and/or made of powdered sugar and milk or lemon juice',
            'both',
            'neither',
            'other'],
            'lawyer': ['boy', 'law', 'both', 'other'],
            'kitty_corner': ['kitty-corner',
            'kitacorner',
            'catercorner',
            'catty-corner',
            'kitty cross',
            'kitty wampus',
            'I can only use "diagonal" for this',
            'I have no term for this',
            'other'],
            'firefly': ['lightning bug',
            'firefly',
            'I use lightning bug and firefly interchangeably',
            'peenie wallie',
            'I have no word for this',
            'other'],
            'verge': ['berm',
            'parking',
            'tree lawn',
            'terrace',
            'curb strip',
            'beltway',
            'verge',
            'I have no word for this',
            'other'],
            'brew_thru': ['brew thru',
            'party barn',
            'bootlegger',
            'beer barn',
            'beverage barn',
            'we have these in my area, but we have no special term for them',
            'I have never heard of such a thing',
            'other'],
            'water_fountain': ['bubbler',
            'water bubbler',
            'drinking fountain',
            'water fountain',
            'other']}

    def create_validation_function(name_of_slot):
        """Function generate our validation functions, since
        they're pretty much the same for each slot"""
        def validate_slot(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
        ) -> Dict[Text, Any]:
            """Validate user input."""

            if value.lower() in self.answers_db()[name_of_slot]:
                # validation succeeded, set the value of the slot to 
                # user-provided value
                return {name_of_slot: value}
            else:
                # set slot to "other" if answer not in original list 
                # of multiple choice answers
                return {name_of_slot: "other"}
        
        return(validate_slot)

    # create validation functions for each of our questions
    validate_second_person_plural = create_validation_function(name_of_slot = "second_person_plural")
    validate_cot_caught = create_validation_function(name_of_slot = "cot_caught")
    validate_rain_sun = create_validation_function(name_of_slot = "rain_sun")
    validate_crawfish = create_validation_function(name_of_slot = "crawfish")
    validate_halloween = create_validation_function(name_of_slot = "halloween")
    validate_sandwich = create_validation_function(name_of_slot = "sandwich")
    validate_side_road = create_validation_function(name_of_slot = "side_road")
    validate_beverage = create_validation_function(name_of_slot = "beverage")
    validate_shoes = create_validation_function(name_of_slot = "shoes")
    validate_highway = create_validation_function(name_of_slot = "highway")
    validate_yard_sale = create_validation_function(name_of_slot = "yard_sale")
    validate_rubbernecking = create_validation_function(name_of_slot = "rubbernecking")
    validate_frosting = create_validation_function(name_of_slot = "frosting")
    validate_lawyer = create_validation_function(name_of_slot = "lawyer")
    validate_kitty_corner = create_validation_function(name_of_slot = "kitty_corner")
    validate_firefly = create_validation_function(name_of_slot = "firefly")
    validate_verge = create_validation_function(name_of_slot = "verge")
    validate_brew_thru = create_validation_function(name_of_slot = "brew_thru")
    validate_water_fountain = create_validation_function(name_of_slot = "water_fountain")

    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Define what the form has to do
            after all required slots are filled"""

        # utter submit template
        dispatcher.utter_message(template="utter_submit")

        return []



class DetectDialect(Action):
    """Detect the users dialect"""

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "detect_dialect"

    def run(self, dispatcher, tracker, domain):
        """place holder method for guessing dialect """
        # let user know the analysis is running
        dispatcher.utter_message(template="utter_working_on_it")

        # get information from the form (maybe)
        bug_slot_info = tracker.get_slot("bug")
        print(bug_slot_info)

        # classify test case
        # TODO: use user input instead of test case
        d, d_classes, dialect_classifier, test_case = ClassifierPipeline.load_data()
        test_case_encoded = ClassifierPipeline.encode_data(test_case, d)
        dialects = ClassifierPipeline.predict_cities(test_case_encoded, dialect_classifier, d)

        # always guess US for now
        return [SlotSet("dialect", dialects)]

class ClassifierPipeline():
    """Load in calssifier & encoders"""

    def name(self) -> Text:
        """Unique identifier of the classfier """

        return "xgboost_softprob"

    def load_data():
        ''' Load in the pretrained model & label encoders.
        '''
        d = load("classifier\label_encoder.joblib.dat")
        d_classes = load("classifier\encoder_classes.joblib.dat")
        dialect_classifier = load("classifier\dialect_classifier.joblib.dat")
        test_case = load("classifier\\test_case.joblib.dat")

        # remove target class from test data
        del test_case["class_target"]

        # update the classes for each of our label encoders
        for key,item in d.items():
            d[key]._classes = d_classes[key]

        return d, d_classes, dialect_classifier, test_case

    def encode_data(input_data, d):
        ''' Encode our input data with pre-trained label encoders.
        '''
        # encode our test data
        test_case_encoded = input_data

        for i, row in input_data.items():
            test_case_encoded[i] = d[i].transform([input_data[i]])

        test_case_encoded = test_case_encoded.apply(lambda x:x[0])

        return test_case_encoded

    def predict_cities(test_case_encoded, dialect_classifier, d):
        ''' Take in encoded data & return top three predicted cities.
        '''
        # convert input data to DMatrix format
        test_case_encoded_d = xgb.DMatrix(test_case_encoded)
        test_case_encoded_d.feature_names =  test_case_encoded.index.tolist()

        # classify using our pre-trained model
        predictions = dialect_classifier.predict(test_case_encoded_d)

        # return the top 3 classes
        top_3 = np.argsort(predictions, axis=1)[ : ,-3 : ]

        cities = d["class_target"].inverse_transform(top_3[0].tolist())

        return cities