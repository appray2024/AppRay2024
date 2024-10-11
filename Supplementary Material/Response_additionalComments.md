# Response to additional comments

**_We fully respect and appreciate reviewers’ constructive comments, and answer the additional comments as follows_** 


## Reviewer_A

* **_Baselines Selection_**: In Table III, Sections V-B&C, UIGuard[13] and AidUI[12] are pioneering research efforts proposing automated methods for detecting dark patterns in single UI screens, unlike earlier (semi-)manual approaches. 

* **_Clarifications on terms_**: (1)We reported micro and macro precision/recall/ F1 scores to comprehensively evaluate our system and baselines, comparing results within the same metrics (i.e.,macro to macro) (Section VI-B, Table III). (2)We will revise the paper to clarify single and multiple UI screens. (3)LinearLayout is a view group element used in Android development that enables linear layouts.(4)A _Task_ refers to an app feature explored during navigation. _Actions_ are interactions taken on the screen to navigate the app. In Android, an _Activity_ represents a concept with multiple UI screens, like a "Settings Activity" that includes screens for notifications or privacy settings.(5)We will clarify in the revision. 

* **_Challenges in exploring UIs_**: Obtaining UIs from apps is challenging due to unpredictable interactions and varying logic. A single action, like clicking back, can lead to different outcomes depending on the app’s design, while pop-ups often block access to elements, hindering exploration. Near-duplicate elements, such as endless scrolling on a shopping app, complicate the process, as clicking everything can trigger loops where new content constantly loads. Heuristics help, but the diversity of UI designs means no single approach covers all cases. Additionally, some screens require specific action sequences to reach, like going through multiple steps to access a payment page in a shopping app. Missing any step disrupts navigation, making it harder to reach the target. 

* **_Resource ID_**:A Resource ID is a unique identifier used in Android development, typically assigned meaningful labels by developers. For example, a left arrow icon might be given the ID navigateUp, reflecting the icon's semantic purpose. However, resource IDs can sometimes lack meaningful labels, such as using a generic ID like checkbox for a CheckBox element. 
  
* **_Interaction Between the device and LLM_**:Given a task, a history of actions (i.e., previous steps for the current task), and the current UI screen information on the device, we first process this data and prompt the LLM for the next action to be performed. Once the action is obtained, we use ADB to execute it on the device and wait for the new screen to load. This process is repeated until we reach the maximum step threshold (i.e., 15) or the LLM returns a stop action, indicating task completion (Section IV-A-1). We will revise Section IV-A-1. 

* **_Experts’ exploration time_**:We acknowledge concerns about exploration time. However, AppRay runs automatically without supervision, aiming to improve human experts' efficiency in detecting dark patterns. Prior research [5, 8, 14] shows 15 minutes is sufficient for human exploration. Extending this time may increase coverage but reduces practical relevance. We'll add this threat in the revision. 



## Reviewer_B, Reviewer_C:Threats&Limitations&Future Works** 

The construction of the AppRay’s datasets pose a potential *internal threat* due to human bias in annotation. To mitigate this, two authors independently annotated the data, thoroughly understanding the taxonomy, and resolved discrepancies through discussions to ensure consistency and accuracy.  

The user study involves only two participants, which introduces the potential *external threat* regarding human bias. To mitigate this, we selected participants with over two years of dark pattern experience, reflecting real-world constraints where limited expert resources are common. 

Future Works: Future research could expand the dataset to include less popular apps and extend detection beyond Android UIs to iOS, desktop, IoT, and AR/VR/XR platforms. Dark pattern mitigation and repair are promising areas for further study, as identifying these patterns is just the first step—eliminating them is crucial. Exploring developers’ perspectives, often overlooked in favor of designers' viewpoints, could offer valuable insights. 

We will add this to the Threats to Validity section in the revision. 


## Reviewer_C 

**_Hierarchy Comparison for Deduplication_** 

We exclude the text attribute when identifying duplicates since UI content often changes dynamically, while element types, resource IDs, and check statuses remain constant. For example, in music apps, element structure remains the same despite changing text. We'll add this explanation to Section IV-A-3 in the revision. 


**_Dataset annotation_** 

_Annotation of dynamic dark patterns_: Dynamic dark patterns span multiple UIs, with a key screen indicating their presence (e.g., an ad in a Nagging Ad). We annotate the ad's location and assign a unique identifier to link the key and preceding screens, showing their sequential relationship. This process mirrors single-screen annotation but includes an additional identifier for dynamic instances. 

_Annotation agreement_: Two authors independently annotated the datasets. Annotator A identified 2,223 instances across 873 UIs, while Annotator B identified 2,183 instances across 701 UIs. Automated matching methods found 1,914 matching annotations (76.8%), with discrepancies discussed. Discrepancies arose from missed instances (due to the large volume of 19,722 UIs) and challenges distinguishing between "Nagging Ads" and "Disguised Ads." To resolve this, we re-labeled 30 instances per type and re-annotated 60 data points, achieving complete agreement. After reviewing and resolving discrepancies, we finalized 2,185 dark pattern instances across 876 UIs. 

We will add these details to Section V-B in the revision. 
