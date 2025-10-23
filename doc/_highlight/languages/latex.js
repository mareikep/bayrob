/*! `latex` grammar compiled for Highlight.js 11.11.1 */
(function(){
  var hljsGrammar = (function () {
    'use strict';

    /*
    Language: LaTeX
    Description: Typesetting system for text documents
    Contributors: Mareike Picklum
    Category: common, markup, scientific
    Website: https://www.latex-project.org/
    */

    function latex(hljs) {
      const COMMENTS = hljs.COMMENT(
        '%',
        '$'
      );

      const COMMAND = {
        className: 'keyword',
        begin: /\\/,
        end: /[^a-zA-Z@]/,
        excludeEnd: true,
        relevance: 10
      };

      const SPECIAL_CHARS = {
        className: 'char.escape',
        begin: /\\[%$&#_{}~^]/,
        relevance: 0
      };

      const ENVIRONMENT_BEGIN = {
        className: 'section',
        begin: /\\begin\{/,
        end: /\}/,
        contains: [
          {
            className: 'title',
            begin: /\{/,
            end: /\}/,
            excludeBegin: true,
            excludeEnd: true
          }
        ]
      };

      const ENVIRONMENT_END = {
        className: 'section',
        begin: /\\end\{/,
        end: /\}/,
        contains: [
          {
            className: 'title',
            begin: /\{/,
            end: /\}/,
            excludeBegin: true,
            excludeEnd: true
          }
        ]
      };

      const BRACES_PARAM = {
        begin: /\{/,
        end: /\}/,
        relevance: 0,
        contains: [
          'self',
          COMMAND,
          SPECIAL_CHARS
        ]
      };

      const BRACKETS_PARAM = {
        className: 'params',
        begin: /\[/,
        end: /\]/,
        relevance: 0,
        contains: [
          COMMAND,
          SPECIAL_CHARS
        ]
      };

      const MATH_INLINE = {
        className: 'formula',
        variants: [
          {
            begin: /\$/,
            end: /\$/
          },
          {
            begin: /\\\(/,
            end: /\\\)/
          }
        ],
        contains: [
          SPECIAL_CHARS,
          COMMAND
        ],
        relevance: 5
      };

      const MATH_DISPLAY = {
        className: 'formula',
        variants: [
          {
            begin: /\$\$/,
            end: /\$\$/
          },
          {
            begin: /\\\[/,
            end: /\\\]/
          }
        ],
        contains: [
          SPECIAL_CHARS,
          COMMAND
        ],
        relevance: 10
      };

      return {
        name: 'LaTeX',
        aliases: ['tex'],
        case_insensitive: false,
        contains: [
          COMMENTS,
          MATH_DISPLAY,
          MATH_INLINE,
          ENVIRONMENT_BEGIN,
          ENVIRONMENT_END,
          COMMAND,
          SPECIAL_CHARS,
          BRACKETS_PARAM,
          BRACES_PARAM
        ]
      };
    }

    return latex;
  })();

  hljs.registerLanguage('latex', hljsGrammar);
})();
