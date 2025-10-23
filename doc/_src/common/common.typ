#include "/_src/common/highlight.typ"
#include "/_src/common/togglestyle.typ"
#include "/_src/common/footer.typ"
#include "/_src/common/nav.typ"

#set document(title: "BayRoB Docs")

#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#show raw.where(block: true): block.with(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
)