from typing import Dict, Text, Any, List, Union, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import SlotSet

class ElicitationForm(FormAction):
    """Example of a custom form action"""

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "elicitation_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["bug", "beverage", "second_person_plural"]

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
                self.from_text()]
        }

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

        # always guess US for now
        return [SlotSet("dialect", "the United States")]
