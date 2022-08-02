# Conventions

A brief note about nomenclature and formatting conventions.

## Branch names

**Please ensure meaningful names**

Core features of the code should be developed in a `feature-*` branch off one of the main
branches of the repository.
Bugfixes should be developed in a `bugfix-*` branch and merged only after review.

If you want to test a certain implementation please create a `test-*` branch. These **will**
be removed later.

Specifically: If you want to test wether a subset of the features or minor fixes you have
developed in a `feature-*` or `bugfix-*` branch are sufficient to fix a problem in the main
(or feature) branch please create a `test-*` branch off the main branch, e.g. `test-abcd` for
a test of modifications to commit `abcd` and import your changes there by merging or cherry-picking.

# General code contribution guideline

The guidelines provided below are intended to ensure a sustainable development and to prevent
contributors (re-)introducing bugs and errors due to unclear code structure.

These rules are not set in stone and can be amended if a convincing argument for a change
is made.

Since the project is in Python it is **imperative** that code formatting guidelines are followed,
since an inconsistency (especially regarding spaces) will immediately result in frustration.

## General rules
  - If your commit changes functionality or interfaces it **must** be
  submitted as a _pull request_, **do not** push your commit directly into the repository!
  - Minor bugfixes _can_ be submitted directly, but should be explained in the commit message!
  - Work in your private fork of the repository on Github and commit changes there first, then
  open a pull request.

## Contributable objects
The following objects can be added to the repository:
  - Source code in C, C++, Fortran, Python, Matlab
  - Data files necessary to run examples
  - Sample configurations (as text files)
  - Output (e.g. `output.csv`) **if and only if** it is to be used as reference for further tests.
  Such a case *must* be documented.
  - Documentation formatted according to Markdown rules.

## NON-contributable objects
To prevent useles data transfers, bloating of the repository and unncessary clean-up operations
the following **must not** be added to the commits to this repository:
  - binary files (any compiled file, e.g., `*.pyc` and library)
  - Files incidental to the compilation process, such as:
    - object files `*.o`
    - debug files `*.d`
    - precompiled headers `*.gch`
  - Image files. With following exceptions
    - EPS/SVG vectorgraphics as those are textual files describing an image
    - Images used to document reference results.
        - Such images should be contributed via a separate pull request and require
          at least one reviewer in addition to the original author.

## Nomenclature in this document
  - oc-token = open close token such as parenthesis `[,]`, `(,)`, `{,}` and `<,>`
  - prefix tag = all characters before an *oc-token*
  - operator = math operator symbols like `+`, `-`, `/` ...
  - regex = regular expression to indicate where spaces must be placed
    - `{...}` = optional definition of `...`
    - `[:...:]` = explicit placement of `...`

## 1. Files and Encoding
  - files **must** be encoded in ASCII
  - LF ( line feed, `\n` , `0x0A` ) **must** be used for new lines
  - Each file should contain a doxygen `@file ...` documentation.
  - `import` directives:
    - order from the most specific/specialized to the most general one
    - objects used from an import **can** be added in a comment but should **at most** be used for stable third party includes.

## 3. Global Rules
  - **use descriptive names**, e.g., `numThreads` instead of `T`.
  - wrap a line after column 80
  - **do not use TABs**, use four (**4**) spaces per indentation level
  - no code alignment (only indentation)
  - names *should* be consistent
  - any new line within an *oc-token* must be preempted by a `\`
  - if there is no content between `{ }`, `< >`, `[ ]` or `( )`
    - the *oc-token* token **must** be on the same line
    - one space between both *oc-tokens* ( regex: `<[:space:]>` )
  - *oc-token* `( )`
    - for function / method signatures
      - `(` is placed after a *prefix tag* without a space before e.g., `method( );`
      - `(` can be followed by a new line ( except: if `( )` empty )
      - `)` is always placed on the same indentation level as the opening token ( except: if `( )` is empty )
    - for expressions e.g., `( a + 1 ) * 5;`
      - can be placed on one line
      - after 80 characters a new line **must** follow
      - after first new line start indentation
    - for function calls e.g., `foo( this );`
      - for one function parameter use one line
      - for more than one function parameter you **may** use one line _but_ you **must** wrap the line
      after column 80
      - it is _advisable_ to place each parameter indented on a new line

## 4. Comments
  - multiline comments
    - begin with `'''` followed by a space ( regex: `'''[:space:]` )
    - end with `'''` on a **new line**
  - single line comments begin with `#` followed by a space ( regex: `#[:space:]` )
  - doxygen inside comments is highly encouraged


## 5. Doxygen Comments
   - oneline comments via `#` should only be used when a brief title alone is descriptive enough (e.g. definition of a member variable)
   - for all other, multiline comments **should** be used for doxygen comments
   - brief ("title")
     - begins with `#`
     - should be a single line comment
   - long description
     - is separated with an empty line
     - is not aligned with the brief line
     - `@author` - attribution (do NOT remove authors!)
     - `@date` - you _should_ add the date the file was last modified

## 6. Operators
  - unary operators are placed directly before or behind the corresponding object e.g.`!enabled`
  - binary operators
    - should be surrounded by spaces e.g., `x = a + b;`
  - after 80 characters a new line must follow
      - use the binary operator as the last code of the previous line
      - the first new line starts a new indentation block

## 7. Function Calls
  - the parameter type should be hinted in the function call `foo( a : float )`
  - if an argument should not to be changed copy it within the function into a local object.
  - for **one** function parameter
    - use one line e.g., `method( 2 );`
    - parameter is surrounded by spaces
  - for **more than one** function parameter place each indented on a new line **if**
  the entire parameter set will exceed 80 columns/characters.
  - `( ... )` are part of the *caller* (see above), no space to that caller


## 8. Function and Method Definitions
  - use of `assert` to control PRE/POST conditions is highly encouraged!
  - objects should be named in a consistent manner in a source file, `camelCase` is preferred
  - function names should be placed on a new line followed by the *oc-token* `(`
  - if *oc-token* `)` is placed on a new line it must be at the same indentation level as `(`
  - code between `{` and `}` is indented
  - the function / method body `{ }` is followed by an empty line

## 9. Code contribution process

Changes should be implemented and tested in a local `clone` of the forked repository.
This allows for a simpler movement of code between development and testing systems.

Once a state ready for review is achieved a pull request should be opened into the `dev` branch of this
repository. After an approval by the reviewer(s) the contribution should be rebased to _squash commits_ 
if the number of individual commits exceeds 5 (this request will be noted in the PR discussion).
The intent is to not bloat the shared development tree.

A review **is required** before the PR can be merged! Should a review not happen within 2 days (48 h) 
_and_ the code pass existing tests then the pull request can be merged by the contributor themselves.

### 9.1 Using different remotes

The general process of development, as delineated above is:
  1. Fork this repository to obtain a fork under your account.
  2. Clone your fork to the development machine.
  3. Implement changes in the local clone according to the rules above.
  4. Push the changes to your fork.
  5. Open a pull request to contribute the changes to this repository.

This process may be amended/overriden by committing directly from the local clone of one's own
fork to _this_ repository. **Such actions should only be taken in agreement with other developers**
as they bear a high potential of messing up the development history, requiring extended fixing.

To commit _from the local clone of a fork_ into the base repository the base repository
must be added as a separate _remote_ repository via:

`git remote add -t dev baseRepo git@github.com:Anton-Le/PhysicsBasedBayesianInference.git`

Here the `baseRepo` is the label the remote repository will be listed under
in the local clone and `dev` is the default branch. The label for the repote repository
is arbitrary.

To update the `dev` branch of the local clone directly from the base repositry, rather than
taking the pathway of fetching the changes into the form in GitHub and then pulling the
changes from the fork into the local copy, one can do the following:

`git checkout dev`
`git pull baseRepo dev`

which will pull the `dev` branch of the base repository into the currently checked-out branch.

Similarly, to push the changes from a branch of the local copy into the same branch of the base
repository, bypassing the forked repository on GitHub, is achieved via:

`git checkout feature-...`
`git push baseRepo`

This will push the changes made to the `feature-...` branch from the local copy directly to the
`feature-...` branch of the base repository.
