## happy path
* greet: howdy
    - utter_greet   <!-- predicted: utter_start_quiz -->
    - utter_start_quiz   <!-- predicted: action_default_fallback -->
    - action_listen   <!-- predicted: action_default_fallback -->
* affirm: sure
    - elicitation_form   <!-- predicted: action_default_fallback -->
    - form{"name": "elicitation_form"}
* form: inform: something   <!-- predicted: deny: something -->
    - slot{"bug": "pill bug"}
    - slot{"beverage": "soda"}
    - slot{"second_person_plural": "y'all"}
    - slot{"cot_caught": "different"}
    - slot{"rain_sun": "I have no term or expression for this"}
    - slot{"crawfish": "crawfish"}
    - slot{"halloween": "I have no word for this"}
    - slot{"sandwich": "sub"}
    - slot{"side_road": " access road"}
    - slot{"shoes": "tennis shoes"}
    - slot{"highway": "highway"}
    - slot{"yard_sale": "yard sale"}
    - slot{"rubbernecking": "rubbernecking"}
    - slot{"frosting": "frosting"}
    - slot{"lawyer": "boy"}
    - slot{"kitty_corner": "kitty-corner"}
    - slot{"firefly": "firefly"}
    - slot{"verge": "verge"}
    - slot{"brew_thru": "brew thru"}
    - slot{"water_fountain": "water fountain"}
    - form{"name": null}
    - utter_slots_values   <!-- predicted: action_default_fallback -->
    - action_listen   <!-- predicted: action_default_fallback -->
* thankyou: cool, thanks
    - utter_noworries   <!-- predicted: action_default_fallback -->
    - action_listen   <!-- predicted: action_default_fallback -->


## no quiz
* greet: hello
    - utter_greet   <!-- predicted: utter_start_quiz -->
    - utter_start_quiz   <!-- predicted: action_default_fallback -->
    - action_listen   <!-- predicted: action_default_fallback -->
* deny: nah
    - utter_noworries   <!-- predicted: action_default_fallback -->
    - action_listen   <!-- predicted: action_default_fallback -->


