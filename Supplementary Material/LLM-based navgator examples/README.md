# Detailed Reasons for GPT successful and failed cases

# Detailed Reasons for GPT successful and failed cases

The general failure reasons for all tasks are related to the missing information in view hierarchy, i.e., the image-based buttons do not have content descriptions, making the semantic of these UI elements unknown and the model can not perform the right actions. This issue is also commonly identified as an accessibility issue~\cite{chen2020unblind}. Although, we surprisingly find that as our tool has the spirit to explore the apps, it will sometimes try to interact with the UI elements without description and eventually finish the task. Adding an icon recognition can mitigate this issue substantially.

In addition to the general successful and failure observations, the performance in T1 (Register account), and T7 (sign out) has a large room to improve. For registration, while our tool can provide the required information, the task still failed because some apps require email/phone verification to finish the registration process. One rare but interesting case is the LiSTNR app.
As seen in below Figure, after our tool provides the right email address, the continue button disappears, exposing a usability issue in the app.
For signing out tasks, our tool fails mostly when it requires to click on the user name rather finding a ``Sign Out'' button/text in the setting page. 


<img src="./examples-listner.pdf" alt="Figure: Usability issue in Listner App"   width="350"/>

