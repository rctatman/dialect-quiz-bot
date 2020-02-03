## happy path
* greet: howdy
    - utter_greet
    - utter_start_quiz
* affirm: sure
    - elicitation_form
    - form{"name": "elicitation_form"}
    - form{"name": null}
    - utter_slots_values
* thankyou: cool, thanks
    - utter_noworries

## no quiz
* greet: hello
    - utter_greet
    - utter_start_quiz
* deny: nah
    - utter_noworries