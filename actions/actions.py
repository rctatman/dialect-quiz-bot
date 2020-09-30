from typing import Dict, Text, Any, List, Union, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import SlotSet, ConversationPaused

from joblib import load
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import sklearn.neighbors
from fuzzywuzzy import process

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
                self.from_text(
                intent="inform")],
            "beverage": [self.from_entity(
                entity="beverage", 
                intent="inform"), 
                self.from_text(
                intent="inform")],
            "second_person_plural": [self.from_entity(
                entity="second_person_plural", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "cot_caught": [self.from_entity(
                entity="cot_caught", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "rain_sun": [self.from_entity(
                entity="rain_sun", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "crawfish": [self.from_entity(
                entity="crawfish", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "halloween": [self.from_entity(
                entity="halloween", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "sandwich": [self.from_entity(
                entity="sandwich", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "side_road": [self.from_entity(
                entity="side_road", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "shoes": [self.from_entity(
                entity="shoes", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "highway": [self.from_entity(
                entity="highway", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "yard_sale": [self.from_entity(
                entity="yard_sale", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "rubbernecking": [self.from_entity(
                entity="rubbernecking", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "frosting": [self.from_entity(
                entity="frosting", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "lawyer": [self.from_entity(
                entity="lawyer", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "kitty_corner": [self.from_entity(
                entity="kitty_corner", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "firefly": [self.from_entity(
                entity="firefly", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "verge": [self.from_entity(
                entity="verge", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "brew_thru": [self.from_entity(
                entity="brew_thru", 
                intent="inform"),
                self.from_text(
                intent="inform")],
            "water_fountain": [self.from_entity(
                entity="water_fountain", 
                intent="inform"),
                self.from_text(
                intent="inform")]
        }

    # validate user answers
    @staticmethod
    def answers_db() -> Dict[str, List]:
        """Database of multiple choice answers"""
        return{"lawyer":["either","other","law","boy"],
        "cot_caught":["different","other","same"],
        "second_person_plural":["other","y'all","yins",
        "you","you 'uns","you all","you guys","you lot",
        "yous, youse"],
        "yard_sale":["car boot","car boot sale",
        "carport sale","garage sale","jumble (sale)",
        "other","patio sale","rummage sale","sidewalk sale",
        "stoop sale","tag sale","thrift sale","yard sale"],
        "verge":["beltway","berm","curb strip",
        "I have no word for this","other","parking",
        "terrace","tree lawn","verge"],
        "sandwich":["baguette","bomber","grinder","hero",
        "hoagie","I have no word for this","Italian sandwich",
        "other","poor boy","sarney","sub"],
        "firefly":["firefly","I have no word for this",
        "I use lightning bug and firefly interchangeably",
        "lightning bug","other","peenie wallie"],
        "crawfish":["craw","crawdad","crawfish","crayfish",
        "crowfish","I have no word for this critter","mudbug","other"],
        "shoes":["gymshoes","I have no general word for this",
        "jumpers","other","runners","running shoes","sand shoes",
        "shoes","sneakers","tennis shoes","trainers"],
        "bug":["basketball bug","centipede","doodle bug",
        "I have no idea what this creature is",
        "I know what this creature is, but have no word for it",
        "millipede","other","pill bug","potato bug","roll-up bug",
        "roly poly","sow bug","twiddle bug","wood louse"],
        "kitty_corner":["catercorner","catty-corner",
        "I can only use \"diagonal\" for this","I have no term for this",
        "kitacorner","kitty-corner","kitty cross","kitty wampus","other"],
        "highway":["a freeway has limited access (no stop lights, no intersections), whereas a highway can have stop lights and intersections",
        "a freeway is bigger than a highway",
        "a freeway is free (i.e., doesn't charge tolls); a highway isn't",
        "expressway","freeway","highway","other","parkway",
        "throughway/thru-way","turnpike"],
        "rain_sun":["fox's wedding","I have no term or expression for this",
        "liquid sun","monkey's wedding","other","pineapple rain","sunshower",
        "the devil is beating his wife","the wolf is giving birth"],
        "frosting":["both","frosting","icing",
        "icing is thinner than frosting, white, and/or made of powdered sugar and milk or lemon juice",
        "neither","other"],
        "side_road":["access road","feeder road","frontage road",
        "gateway","I've never heard of this concept","other",
        "service road","we have them but I have no word for them"],
        "water_fountain":["bubbler","drinking fountain","other","water bubbler",
        "water fountain"],
        "beverage":["cocola","coke","dope","fizzy drink",
        "lemonade","other","pop","soda","soft drink","tonic"],
        "rubbernecking":["curiosity delay","gapers' block",
        "gapers' delay","gawk block","I have no word for this",
        "Lookie Lou","other","rubberneck","rubbernecking",
        "rubbernecking is the thing you do, not the traffice jam"],
        "halloween":["cabbage night","devil's eve","devil's night",
        "gate night","goosy night","I have no word for this",
        "mischief night","other","trick night"],
        "brew_thru":["beer barn","beverage barn","bootlegger","brew thru",
        "I have never heard of such a thing","other","party barn",
        "we have these in my area, but we have no special term for them"]}

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
                # find the closest answer by some measure (edit distance?)
                choices = self.answers_db()[name_of_slot]
                answer = process.extractOne(value.lower(), choices)

                # check to see if distnace is greater than some threshold
                if answer[1] < 45:
                    # if so, set slot to "other"
                    return {name_of_slot: "other"}
                else:
                    return {name_of_slot: answer[0]}
        
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
    validate_bug = create_validation_function(name_of_slot = "bug")

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
        """Unique identifier of the action"""

        return "detect_dialect"

    @staticmethod
    def slot_key_db() -> Dict[str, List]:
        """Database of slot values & 
        corresponding questions"""

        return {'q50': 'second_person_plural',
            'q28': 'cot_caught',
            'q80': 'rain_sun',
            'q66': 'crawfish',
            'q110': 'halloween',
            'q64': 'sandwich',
            'q90': 'side_road',
            'q105': 'beverage',
            'q73': 'shoes',
            'q79': 'highway',
            'q58': 'yard_sale',
            'q107': 'rubbernecking',
            'q94': 'frosting',
            'q14': 'lawyer',
            'q76': 'kitty_corner',
            'q65': 'firefly',
            'q60': 'verge',
            'q118': 'brew_thru',
            'q103': 'water_fountain'}

    def run(self, dispatcher, tracker, domain):
        """get dialect classification """
        # let user know the analysis is running
        # dispatcher.utter_message(template="utter_working_on_it")

        # get information from the form & format it
        # for encoding
        slot_question_key = self.slot_key_db()
        formatted_responses = pd.Series(index = slot_question_key.keys())

        for index, value in formatted_responses.items():
            formatted_responses[index] = tracker.get_slot(slot_question_key[index])

        # classify test case
        dialects = ClassifierPipeline_knn().get_top_3_knn(formatted_responses)

        state_1, state_2, state_3 = dialects

        dialects = f"The state that mostly closely matches your language use is {state_1}, followed by {state_2} and {state_3}"

        # always guess US for now
        return [SlotSet("dialect", dialects)]

class ClassifierPipeline_knn():
    """Load in classifier & encoders"""

    def name(self) -> Text:
        """Unique identifier of the classfier """

        return "5knn_state"

    def encode_answers(self, input_data):
        '''Reads in the sample encoded data w/ correct columns and 
        converts input data to the same format'''
        # read in empty dataframe with correct columns
        encoding_sample = pd.read_csv("empty_data_stucture.csv").iloc[:, 3:]
        
        # transpose input data
        input_data = pd.DataFrame(input_data).transpose()

        # one hot encode input data & standardize column names
        encoded_input_data = pd.get_dummies(input_data)
        encoded_input_data.columns = [col.replace(' ','_') for col in encoded_input_data.columns]

        # encode it
        encoded_data = encoding_sample.align(encoded_input_data,
        join = "left", axis = 1)

        # convert na's to 0 (since we're one hot encoding)
        encoded_data = encoded_data[1].fillna(0)
        
        return(encoded_data)

    def get_top_3_knn(self, data):
        '''Read in the knn model and apply it to correctly formatted sample data'''
        # read in model
        state_knn = load("state_level_knn.joblib")

        # encode input data
        encoded_data = self.encode_answers(input_data = data)

        pred = state_knn.predict_proba(encoded_data)
        top_3 = np.argsort(pred, axis=1)[ : ,-3 : ]
        results = [state_knn.classes_[i] for i in top_3]

        return(results[0].tolist())


class PauseConversation(Action):
    """Pause the conversation so the the 
    assistant won't respond"""

    def name(self) -> Text:
        """Unique identifier of the action"""

        return "pause_conversation"

    def run(self, dispatcher, tracker, domain):

        return [ConversationPaused()]
