/* _src/pages/users.typ */
#include "/_src/common/common.typ"
#import "/_src/mod.typ": *
#set heading(numbering: "1.")

//=============================================================================

= For Users

BayRoB provides a web interface that allows to

 - do things
 - and even do more things

//=============================================================================

== Usage

=== Reasoning

BayRoB reasoning can be executed by defining a query...


Clicking the `Query`-button will trigger the reasoning process.

#html.img(src: "_assets/img/webinterface_query.png", width: 400, alt: "Web interface: Query")

*The BayRoB Query window*

The results of the reasoning process will be shown in two ways. The first one is a visualization ...
(see :num:`Fig. =webif-hyps`: :ref:`webif-hyps`). The distributions plots can be downloaded as ``.svg`` files.

#html.img(src: "_assets/img/webinterface_query.png", width: 400, alt: "Web interface: Query")

*The visualization of the query results allows to compare the ground truth with the computed (posterior) distributions.*

//=============================================================================

=== Plan Refinement

#html.img(src: "_assets/img/webinterface_search.png", width: 400, alt: "Web interface: Search")
*The BayRoB Search window*

//=============================================================================

