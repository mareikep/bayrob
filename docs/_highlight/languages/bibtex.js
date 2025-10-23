/*! `bibtex` grammar compiled for Highlight.js 11.11.1 */
(function(){
  var hljsGrammar = (function () {
    'use strict';

    /*
    Language: BibTeX
    Description: Bibliography format for LaTeX documents
    Contributors: Mareike Picklum
    Category: common, markup, scientific
    Website: https://www.bibtex.org/
    */

    function bibtex(hljs) {
      const COMMENTS = hljs.COMMENT(
        '%',
        '$'
      );

      const ENTRY_TYPE = {
        className: 'keyword',
        begin: /@(article|book|booklet|conference|inbook|incollection|inproceedings|manual|mastersthesis|misc|phdthesis|proceedings|techreport|unpublished)/,
        relevance: 10
      };

      const CITE_KEY = {
        className: 'symbol',
        begin: /\{/,
        end: /,/,
        excludeBegin: true,
        excludeEnd: true,
        relevance: 10
      };

      const FIELD_NAME = {
        className: 'attr',
        begin: /[a-zA-Z]+\s*=/,
        end: /=/,
        excludeEnd: true,
        relevance: 5
      };

      const STRING_VALUE = {
        className: 'string',
        variants: [
          {
            begin: /"/,
            end: /"/,
            contains: [hljs.BACKSLASH_ESCAPE]
          },
          {
            begin: /\{/,
            end: /\}/,
            contains: ['self']
          }
        ]
      };

      const NUMBER_VALUE = {
        className: 'number',
        begin: /\b\d+\b/,
        relevance: 0
      };

      const ENTRY = {
        begin: /@[a-zA-Z]+\s*\{/,
        end: /\}/,
        contains: [
          ENTRY_TYPE,
          CITE_KEY,
          FIELD_NAME,
          STRING_VALUE,
          NUMBER_VALUE,
          COMMENTS
        ],
        relevance: 10
      };

      return {
        name: 'BibTeX',
        aliases: ['bib'],
        case_insensitive: true,
        contains: [
          COMMENTS,
          ENTRY
        ]
      };
    }

    return bibtex;
  })();

  hljs.registerLanguage('bibtex', hljsGrammar);
})();