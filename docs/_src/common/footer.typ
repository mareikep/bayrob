/* _src/common/footer.typ */
#import "/_src/mod.typ": css, struct

#import struct: *

// Define and style the footer.

#css.elems((
  footer: (
    // Follows the bottom of the page
    position: "fixed",
    bottom: 0,
    // Same width as body, see `_assets/css/global.css`
    width: 100%,
    max-width: "1200px",
  )
))

#html.footer[
  #box(width: "100%", inset: 1pt, radius: 3pt, fill: "var(--dk-gray2)")[
    #box(inset: (x: 5mm),
      text(size: 10pt)[
        Last build: #datetime.today().display()
      ]
    )
    #box(inset: (x: 5mm),
      text(size: 10pt)[
        © Copyright 2025 #link("mailto:mareikep@cs.uni-bremen.de")[Mareike Picklum] 
      ]
    )
    #box(inset: (x: 5mm),
      text(size: 10pt)[
        #html.a(href: "https://github.com/mareikep/bayrob", {
          html.img(src: "_assets/img/github.svg", width: 20)
        })
      ]
    )
  ]
]
