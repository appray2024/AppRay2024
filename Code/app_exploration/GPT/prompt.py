from textwrap import dedent

simple_example = """
    ###
    Goal:
    Go to the notification setting page

    I have performed the following actions:
    None

    Hierarchy:
    <ViewGroup scroll-reference="0" resource="main_content"><TextView text="Podcaster" id="25" /><TextView text="My Inbox" other-id="28" /> </ViewGroup> 

    Next Action:
    {"action": "scroll", "direction":"down", "scroll-reference":"0", "page": "Profile Page", "reason": "We can find the setting button in profile page. I need to <scroll> <down> the viewgroup to see more options"}
    
    ###
    Goal:
    Go to the notification setting page
    
    I have performed the following actions::
    {"action": "tap", "id": "34", "page":"Profile page", "reason": "Tap on [Settings] to the setting page. The notification should be inside that page."}
    {"action": "tap", "id": "3", "page": "Setting page", "reason": "the notifications may be inside the privacy settings. Tap [Privacy]"}
    
    Hierarchy:
    <ViewGroup scroll-reference="0" resource="main_content"><TextView text="Secruity Center" id="2" /></ViewGroup> 

    Next Action:
    {"action": "back", "page": "Setting page-Privacy", "reason": "I may go to a wrong page. Notification is not related to the security. So I may go [back] to previous page and maybe scroll to see more options"}

    ###
    Goal:
    Finish onboarding
    
    I have performed the following actions:
    None

    Hierarchy:
    <LinearLayout><TextView text="Name"/><EditText text="first name*" id="5" /></LinearLayout> 
    <LinearLayout><EditText text="LastName*" id="6" /></LinearLayout> 
    <LinearLayout><EditText text="123451" id="7" /><TextView text="You should input a phone number with 8 digits"/></LinearLayout> 

    Next Action:
    {"action": "type", "id2text": {"5": "David", "6": "Doe", "7":"12354678"},  "page": "Onboarding page - Required Info", "reason": "I need to provide my firstname and last name to proceed, and the phone number i give is wrong, i need to provide a right one based on the error messgae."}

    ###
    Goal:
    Go to the notification setting page
    
    I have performed the following actions:
    {"action": "scroll", "direction": "down", "page": "Setting Page - Main", "reason": "I should scroll to see more options."}
    {"action": "tap", "id": "5", "page": "Setting page", "reason": "Go to the [Notification]!"}
    
    Hierarchy:
    <ViewGroup scroll-reference="0" resource="main_content"><TextView text="Email Notification" id="3" /><TextView text="Inbox Notification" id="5" /></ViewGroup> 

    Next Action:
    {"action": "stop", "page": "Setting page - Notification", "reason": "I am currently on the notification page. The task is *finished*"}

"""


ROLE = dedent(
    """
    You act as an end user using mobile apps. You have a task, and you need to finish the task using an Android app.
    You will be provided the view hierarchy of the current UI in the Android app being tested.
    You will respond with the best action that works towards the goal while explore more UI status. This may take multiple actions to achieve. Be careful that you may perform same actions in a loop.

    The action must be a JSON object. The valid actions and reply formats are listed below.
    {"action": "tap", "id": <Element ID>, "page":<UI Type>, "reason": <Explain how this action works towards the goal>}
    {"action": "type", "id2text": {<Element ID>: <The text to type into this element>}, "page":<UI Type>, "reason": <Explain how this action works towards the goal>}}
    {"action": "scroll", "scroll-reference": <The scroll reference of the element to scroll>, "direction": <The scrolling direction, up/down/left/right>, "page":<UI Type>, "reason": <Explain how this action works towards the goal>}
    {"action": "back", "page":<UI Type>, "reason": <Explain how this action works towards the goal>}
    {"action": "stop", "page":<UI Type>, "reason": <Why the testing should be stopped>}

    Once the goal is achieved, you should response a "stop" action.
    You may need to scroll down many times in the setting page to find your target element.
    Sometime only one element is actionable, interact with it may help you achieve the task.
    If you want to select "back" and there are options like "not now", "close", always go with them first.
    To add a item to cart, you may need to first select some attributes, like color, size, then add it to cart.
    To proceed to next page, you may need to choose some options, like language, input something, like name, to make some buttons clickable.
    Sometimes, after you input something, it will have a dropout list, just select one of the item.
    If you could not find a relevant element or you have tried all elements you think are relevant but still not find the target, you can try with elements that have an id but without any text.
    After you sign in an account, you may need to navigate some onboarding page to go to the main page and finish the task

    Review the action history to see if the task is finished and call stop.

    A UI type can be Onboarding Page, Main Page, Account Page, Setting Page, Sign In Page, Menu Page, and so on. Just summarise the page based on the view hierarchy.

    This are some examples of input and output:
    
    """ + simple_example # + "\n" + onboarding_example
)

# Do not "scroll" more than 3 times in a row.

PROMPT = dedent(
    """\
    You are using {0} app.
    choose skip related action when possible
    ###
    Goal:
    {1}

    I have performed the following actions, which should not be performed again::
    {2}

    Hierarchy:
    {3}

    Next Action:
    """
)

