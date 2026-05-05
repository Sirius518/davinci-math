from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_TRANSLATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"translate the (?:above )?text into english",
        r"retain the original text's line breaks",
        r"output the translation result directly",
        r"untranslated (?:part|text|portion)",
        r"the translation is the same as the original",
        r"translated as requested",
        r"preserv(?:e|ing) the original (?:text's )?format",
    ]
]

_MATH_LATEX_RE = re.compile(
    r"\$.*?\$"
    r"|\\(?:frac|sqrt|int|sum|prod|lim|log|sin|cos|tan|binom|boxed|begin\{)"
    r"|\\(?:triangle|angle|pi|theta|alpha|beta|gamma|infty|cdot|times|div)"
    r"|\\(?:geq|leq|neq|approx|equiv|pmod|mathbb|mathrm)"
)

_MATH_INSTRUCTION_RE = re.compile(
    r"\b(?:solve|find|determine|evaluate|prove|simplify|calculate|compute|"
    r"show that|how many|what is the (?:value|sum|product|remainder|probability))\b",
    re.IGNORECASE,
)

_MATH_OBJECT_RE = re.compile(
    r"\b(?:equation|inequality|integer|polynomial|probability|triangle|circle|"
    r"rectangle|polygon|matrix|determinant|derivative|integral|limit|"
    r"sequence|series|permutation|combination|divisor|prime|modulo|"
    r"hypotenuse|midpoint|diameter|radius|vertex|vertices|"
    r"arithmetic|geometric|binomial|factorial|logarithm)\b",
    re.IGNORECASE,
)

_NON_MATH_CODE_RE = re.compile(
    r"\breact\b|\btypescript\b|\bjavascript\b|"
    r"\bpython (?:program|script|code|class|function)\b|\bcode snippet\b|"
    r"\bgenerate an executable\b|\bfunction that (?:takes|returns|accepts)\b|"
    r"\bhtml\b|\bcss\b|\bjson\b|\bxml\b|\bapi\b|\bhttp\b|"
    r"\bcomponent\b|\bwebpack\b|\bnpm\b|\bpip install\b|"
    r"\bclass \w+\s*[({]|"
    r"\bdef \w+\s*\(|"
    r"\bimport \w+|"
    r"\bconsole\.log\b|\bprint\s*\(|"
    r"\brust\b|\bf#\b|\bc\+\+\b|\bbash\b|\bshell\b|\bsql\b|\bdatabase\b|"
    r"\bpostgresql\b|\bmysql\b|\bdocker\b|\bkubernetes\b|"
    r"\bcompetitive programming\b|\bimplementation in\b|"
    r"\bcode block\b|\bcode review\b|"
    # --- aggressive "write/create/design/build/construct/implement" + ANY object ---
    r"\bwrite (?:a |an )?(?:program|script|function|class|method|code|paper|essay|poem|"
    r"story|report|letter|article|blog|email|proposal|review|summary|speech|song|"
    r"paragraph|prompt|query|test|unit test|VBA|macro|regex|sql|rule|policy|"
    r"specification|contract|recipe|tutorial|guide|manual|dialogue|outline|"
    r"hypothesis|thesis|dissertation|critique|memo|note|message|notification|"
    r"announcement|invitation|press release|newsletter|whitepaper)\b|"
    r"\bcreate (?:a |an )?(?:class|program|function|script|method|table|chart|"
    r"dashboard|diagram|visualization|model|pipeline|workflow|template|"
    r"spreadsheet|database|schema|index|view|trigger|procedure|API|endpoint|"
    r"component|module|package|library|framework|interface|service|"
    r"application|website|page|form|survey|quiz|test|report|presentation|"
    r"infographic|poster|flyer|brochure|banner|logo|mockup|wireframe|"
    r"prototype|storyboard|animation|video|playlist|schedule|plan|"
    r"roadmap|strategy|budget|forecast|persona|profile|portfolio)\b|"
    r"\bdesign (?:a |an )?(?:class|system|architecture|schema|database|API|"
    r"interface|layout|page|form|template|component|module|service|"
    r"pipeline|workflow|algorithm|data structure|protocol|pattern|"
    r"experiment|study|survey|curriculum|syllabus|lesson plan)\b|"
    r"\bbuild (?:a |an )?(?:class|function|system|application|website|"
    r"dashboard|pipeline|model|API|service|module|tool|bot|agent|"
    r"crawler|scraper|parser|compiler|interpreter|simulator|emulator)\b|"
    r"\bconstruct (?:a |an )?(?:class|function|program|query|regex|"
    r"table|view|schema|pipeline|workflow|model|proof|argument)\b|"
    r"\bimplement (?:a |an )?(?:function|class|method|algorithm|system|"
    r"module|component|service|API|interface|protocol|pipeline|"
    r"solution|feature|handler|middleware|decorator|observer|factory)\b|"
    # OJ / competitive-programming format
    r"\bExpected Time Complexity\b|\bExpected Auxiliary Space\b|"
    r"\bInput Format\b|\bOutput Format\b|\bSample Input\b|\bSample Output\b|"
    r"\bstdin\b.*\bstdout\b|"
    # explicit coding instructions
    r"\bsolve the following coding problem\b|"
    r"\bsolve the following problem using\b.*\b(?:python|java|c\+\+|rust|go|lisp)\b|"
    r"\bconvert this code\b|\btranslate this code\b|"
    r"\brefactor\b.*\bcode\b|\bdebug\b.*\bcode\b|"
    r"\bdata structure\b|\blinked list\b|\bbinary tree\b|\bhash map\b|"
    r"\bsorting algorithm\b|\btime complexity\b|\bspace complexity\b|"
    r"\bunit test\b|\btest case\b|\berror handling\b|"
    # programming languages / frameworks
    r"\bjava\b(?!script)|\bruby\b|\bscala\b|\bswift\b|\bkotlin\b|\bperl\b|\bphp\b|"
    r"\bdjango\b|\bflask\b|\bspring\b|\bangular\b|\bvue\b|\bnextjs\b|"
    r"\btensorflow\b|\bpytorch\b|\bscikit\b|"
    # generic programming actions / types
    r"\breturn (?:a |the )?(?:list|dict|array|string|tuple|boolean)\b|"
    r"\biterable\b|\bgenerator\b|\bdecorator\b|\bcallback\b|"
    r"\btype hints?\b|\bedge cases?\b|"
    # fenced code blocks
    r"```(?:python|java|javascript|typescript|ruby|php|cpp|c\+\+|rust|go|"
    r"swift|kotlin|scala|sql|bash|shell|csharp|c#|erlang|haskell|lua|"
    r"perl|r\b|matlab|objective-c|dart|groovy|elixir|clojure|fortran|"
    r"assembly|vb|visual basic|powershell|zig|nim|ocaml|racket|scheme|"
    r"lisp|prolog|cobol|ada|fsharp|vhdl|verilog|tcl|awk|sed)\b|"
    # code translation / porting
    r"\bport this code\b|\btranslate this (?:code|snippet)\b|"
    r"\bconvert this (?:code|snippet)\b|"
    # predict output of code
    r"\bpredict the output\b|"
    r"\bgiven (?:the |this )?code\b.*\boutput\b|"
    # "Implement a <language> program/class/function"
    r"\bImplement a (?:Python|Java|C\+\+|Rust|Go|Ruby|PHP|Swift|Kotlin) "
    r"(?:program|class|function|method|script)\b|"
    # "Create a <language> program"
    r"\bCreate a (?:Python|Java|C\+\+|Rust|Go) (?:program|script|class)\b|"
    # code modification instructions
    r"\bmodify (?:a |the )?code\b|\badapt(?:ed)? code\b|"
    r"\breturn the adapted code\b|"
    r"\bretrieve and repeat the exact described function\b|"
    r"\bcode context\b|\btool output\b|\bcli transcript\b|"
    r"\bdiff --git\b|\bdeleted file mode\b|"
    r"\bterminal output\b|\bpatch file\b|"
    # "function 'stuff'" / function-specific code puzzles
    r"\bfunction '?\w+'?\s*,?\s*which manipulates\b",
    re.IGNORECASE,
)

_NON_MATH_GENERAL_RE = re.compile(
    r"\b(?:summary|summarize|email|recipe|story|essay|paragraph|article|"
    r"recommend|career|manager|partner|charity|food bank|stage fright|"
    r"social media|marketing|customer service|cover letter|resume|"
    r"roleplay|fanfic|storytelling|creative writing|"
    r"describe the landscape|how are you feeling|"
    r"remote workers?|code reviews?|best practice|"
    r"seo|keywords for instagram|amazon store|"
    r"law school|guitar|customers?|products?|"
    r"edit (?:this|the) (?:text|paragraph|sentence)|"
    r"your (?:entire )?response should|"
    r"answer the following question\b.*\b(?:capable|classify)|"
    # lifestyle / advice / creative
    r"workout|exercise routine|yoga|"
    r"grad school|university admission|"
    r"youtube channel|podcast|blog post|"
    r"etsy|shopify|e-commerce|"
    r"instagram|tiktok|twitter|facebook|"
    r"breaking news|journalism|"
    r"car enthusiast|car restoration|"
    r"real estate|stock market|"
    r"cooking|ingredient|"
    r"travel guide|travel itinerary|"
    r"fashion|beauty|makeup|"
    r"mental health|therapy|counseling|"
    r"restructuring|revenue.*(?:AI|machine learning|business)|"
    r"codename|project release|"
    r"transitional phrase|improve (?:this|the) sentence|"
    r"rewrite (?:this|the) (?:text|sentence|paragraph)|"
    r"paraphrase|proofread|grammar|"
    r"what (?:are|is) (?:the )?(?:best|most important) (?:factors?|tips?|ways?)|"
    r"how (?:can|should) (?:I|we|a company)|"
    r"launch (?:a|an|my) (?:channel|platform|business|startup)|"
    r"presentation tips|slide deck|"
    r"job (?:search|interview|application)|"
    # general tasks / instructions
    r"make a note for|write a self introduction|"
    r"sustainable construction|novel.*material|"
    r"explain the characteristics of|"
    r"ratio scale of measurement|"
    r"love language|marriage counselor|"
    r"youtube video|youtube channel|movie titles|video tags|"
    r"annotation platform|feature request|product requirement|"
    r"interrogation-style technique|"
    r"traduit en anglais|translate into english|"
    r"family member to open up|"
    r"what is one nation conservatism|"
    # --- aggressive: "write a ..." broad non-math ---
    r"write (?:a |an |the )?(?:paper|essay|report|letter|email|blog|post|"
    r"speech|song|poem|story|review|summary|script|dialogue|memo|note|"
    r"invitation|announcement|press release|newsletter|whitepaper|"
    r"proposal|contract|policy|manual|guide|tutorial|critique|thesis|"
    r"dissertation|outline|hypothesis|message|notification|brochure|"
    r"flyer|poster|introduction|conclusion|abstract|biography|profile|"
    r"description|response|answer|reply|comment|feedback|testimonial|"
    r"caption|tagline|slogan|headline|title|subtitle|blurb|excerpt|"
    r"foreword|preface|afterword|appendix|acknowledgement|dedication|"
    r"epigraph|glossary|bibliography|index|syllabus|lesson plan|"
    r"curriculum|case study|analysis report|research proposal|"
    r"business plan|marketing plan|project plan|grant proposal|"
    r"cover letter|resignation letter|recommendation letter|"
    r"thank you note|condolence message|congratulation message|"
    r"apology letter|complaint letter|reference letter)\b|"
    # --- role / persona assignment (aggressive) ---
    r"(?:your|my) role is\b|you are acting as\b|"
    r"(?:imagine|pretend|assume) you (?:are|were)\b|"
    r"act as (?:a |an |the )?\w+|"
    r"play the role of\b|"
    r"respond (?:as|like) (?:a |an )\w+|"
    r"you(?:'re| are) (?:a |an )(?:helpful|friendly|creative|expert|professional|senior|experienced)\b|"
    # --- broad NLP / text tasks ---
    r"classify (?:the |this |these )?(?:text|sentence|review|comment|tweet|post|document)|"
    r"sentiment (?:analysis|classification)|"
    r"named entity recognition|"
    r"text classification|"
    r"topic modeling|"
    r"extract (?:keywords?|entities|topics|information|data) from|"
    r"generate (?:a |an )?(?:title|headline|caption|summary|tagline|slogan|description|"
    r"name|username|password|hashtag|keyword|tag)\b|"
    # --- instruction-following / format-control (non-math) ---
    r"respond in (?:exactly )?\d+ (?:words?|sentences?|paragraphs?|bullets?|points?)|"
    r"keep (?:your |the )?(?:response|answer|reply) (?:under|within|to) \d+|"
    r"use (?:only |exactly )?\d+ (?:words?|sentences?|paragraphs?)|"
    r"in (?:no more|at most|at least|exactly) \d+ (?:words?|sentences?)|"
    r"limit (?:your |the )?(?:response|answer|output) to\b|"
    r"format (?:your |the )?(?:response|answer|output) as\b|"
    r"provide (?:your |the )?(?:response|answer) in (?:bullet|numbered|list|table) form|"
    # --- conversational / chitchat ---
    r"how are you\b|what do you think about\b|"
    r"tell me (?:about yourself|a joke|something interesting|a fun fact)|"
    r"what(?:'s| is) your (?:name|opinion|favorite|favourite)|"
    r"do you (?:like|enjoy|prefer|have|know|think)\b|"
    # --- broad advice / how-to non-math ---
    r"(?:give|provide|offer|share) (?:me |us )?(?:some |a few |several )?(?:tips?|advice|suggestions?|recommendations?|ideas?|examples?)\b.*"
    r"(?:for|on|about|to|regarding) (?:my|our|the|a|an|your)\b|"
    r"(?:help|assist) me (?:with |in )?(?:writing|drafting|composing|preparing|creating|making|designing|planning|organizing|managing)\b|"
    # --- specific non-math domains (expanded) ---
    r"legal (?:advice|opinion|analysis|document|contract|brief|case|precedent)|"
    r"(?:medical|clinical|health) (?:advice|diagnosis|treatment|symptom|condition)|"
    r"(?:financial|investment|tax|accounting) (?:advice|analysis|plan|strategy|report)|"
    r"political (?:science|analysis|opinion|debate|party|campaign)|"
    r"(?:history|historical) (?:event|figure|period|analysis|context)|"
    r"(?:philosophy|philosophical) (?:argument|analysis|perspective|thought)|"
    r"(?:psychology|psychological) (?:analysis|theory|concept|experiment)|"
    r"(?:geography|geographical) (?:feature|location|region|country|city)|"
    r"(?:music|musical) (?:theory|composition|arrangement|analysis|instrument)|"
    r"(?:art|artistic) (?:style|movement|technique|criticism|analysis|medium)|"
    r"(?:literature|literary) (?:analysis|criticism|device|technique|genre|movement)|"
    r"(?:religion|religious) (?:belief|practice|text|tradition|doctrine)|"
    r"(?:sports?|athletic) (?:training|strategy|analysis|statistics|rule)|"
    r"(?:nutrition|dietary) (?:advice|plan|requirement|supplement|intake)|"
    r"(?:agriculture|farming) (?:technique|practice|crop|soil|irrigation)|"
    r"(?:architecture|architectural) (?:style|design|plan|feature|element)|"
    r"(?:engineering) (?:design|analysis|system|process|specification)|"
    r"(?:environmental|ecology|ecological) (?:impact|analysis|conservation|issue)|"
    r"(?:economic|economics) (?:analysis|theory|policy|model|indicator)|"
    r"(?:sociology|sociological) (?:theory|analysis|concept|study|research)|"
    # --- explicitly non-math "explain" tasks ---
    r"explain (?:the |a )?(?:difference|concept|idea|theory|history|meaning|"
    r"significance|purpose|importance|impact|effect|cause|reason|"
    r"process|mechanism|principle|law|rule|regulation|policy|"
    r"advantage|disadvantage|benefit|drawback|pro|con|"
    r"role|function|responsibility|duty)\s+(?:of|between|behind|in)\b|"
    # --- language / translation tasks ---
    r"translate (?:this|the|these|following)?\s*(?:text|sentence|paragraph|word|phrase|"
    r"document|page|passage|excerpt|quote)\b|"
    r"(?:from|into|to) (?:english|french|spanish|german|chinese|japanese|korean|"
    r"arabic|russian|portuguese|italian|dutch|hindi|bengali|thai|turkish|"
    r"vietnamese|polish|czech|swedish|danish|norwegian|finnish|greek|hebrew|"
    r"romanian|hungarian|indonesian|malay|tagalog|swahili|persian|urdu)\b|"
    # --- debate / opinion / ethical tasks ---
    r"argue (?:for|against|that)\b|"
    r"what are the (?:pros? and cons?|advantages? and disadvantages?)\b|"
    r"(?:is it|should we|should I|do you think) (?:ethical|moral|right|wrong|fair|just)\b|"
    r"discuss (?:the |whether |if )\b)\b",
    re.IGNORECASE,
)

_NON_MATH_STEM_RE = re.compile(
    r"\b(?:chemical|chemistry|reaction|synthesis|compound|molar mass|titrat|"
    r"solvolysis|epoxidation|hydrogenation|decarboxylation|nmr|hplc|ftir|"
    r"spectroscopy|chromatograph|distillation|anhydride|nucleophil|electrophil|"
    r"enantiomer|diastereomer|stereoisomer|oxidation state|"
    r"electron|quantum|photon|wavelength|doppler|gravitational|"
    r"angular momentum|spin|magnetic|electromagnetic|"
    r"nuclear|fusion reactor|fission|radioactiv|"
    r"biological|organism|protein|enzyme|gene|genome|dna|rna|"
    r"mutation|allele|phenotype|genotype|cytochrome|"
    r"medical|clinical|diagnos|symptom|patient|biopsy|"
    r"pharmacolog|drug|dosage|antenatal|"
    r"pendulum|friction|torque|velocity|acceleration|"
    r"thermodynamic|enthalpy|entropy|gibbs free energy|clapeyron|"
    r"refractive index|reflectivity|diffraction|"
    r"fluid dynamics|viscosity|reynolds|bernoulli|"
    r"circuit|resistor|capacitor|inductor|impedance|"
    r"permeability|permittivity|dielectric|"
    # physics – expanded
    r"electric field|magnetic field|Lorentz|"
    r"Maxwell.*equation|Schrodinger|"
    r"special relativity|general relativity|"
    r"Newton(?:'s)? (?:law|second law|third law)|"
    r"projectile|free ?fall|kinetic energy.*potential energy|"
    r"electric potential|"
    r"wavefunction|quantum number|orbital.*electron|"
    r"spectral (?:slope|line|emission)|"
    r"conducting sphere|capacitance|"
    r"EMF.*induced|Faraday|"
    r"redshift|absorption.*spectrum|"
    # chemistry – expanded
    r"molecular weight|molarity|concentration.*(?:M|mol/L)|"
    r"acid[- ]base|neutralization|"
    r"hydrolysis|precipitat|saturated solution|"
    r"K_?sp|solubility product|"
    r"reaction rate|rate constant|activation energy|"
    r"lewis (?:acid|base|structure)|VSEPR|"
    r"hybridization.*(?:sp|sp2|sp3)|"
    r"halogen|halide|pseudohalide|"
    r"phenol|sodium phenate|"
    # biology / medicine – expanded
    r"CRISPR|sgRNA|gene knockout|"
    r"Hi-C|enhancer-promoter|chromatin|"
    r"antibody|antigen|immuno|"
    r"histopathology|carcinoma|tumor|"
    r"breast cancer|BRCA|"
    r"perineal|obstetric|"
    r"diffusion capacity|lung condition|"
    r"renal|hepatic|cardiac|"
    r"DNA probe|nucleotide.*polymorphism|"
    r"biological (?:sample|marker|pathway)|"
    r"neutrino|quark|Higgs|"
    r"Boltzmann.*distribution|"
    r"ionization.*(?:energy|constant)|"
    r"pH.*(?:solution|buffer))\b",
    re.IGNORECASE,
)

PROGRAMMING_PREFIXES = (
    "Generate an executable",
    "Write a program", "Write a function", "Write a script", "Write a class",
    "Write a method", "Write code", "Write a code", "Write the code",
    "Create a function", "Create a class", "Create a program", "Create a script",
    "Build a function", "Build a program", "Build a system", "Build an app",
    "Implement a function", "Implement a class", "Implement a program",
    "Design a system", "Design a class", "Design an algorithm",
    "Develop a function", "Develop a program", "Develop a system",
    "Construct a function", "Construct a program",
    "Code a function", "Code a program",
    "Write a paper", "Write an essay", "Write a story", "Write a letter",
    "Write a poem", "Write a song", "Write a speech", "Write a report",
    "Write a review", "Write a blog", "Write an email", "Write a proposal",
    "Write a summary", "Write a paragraph", "Write a dialogue",
    "Write a memo", "Write a note", "Write a message", "Write an article",
    "Draft a letter", "Draft a report", "Draft an email", "Draft a proposal",
    "Draft a memo", "Draft a plan", "Draft a response", "Draft a policy",
    "Compose a letter", "Compose an essay", "Compose a poem", "Compose a story",
    "Compose a song", "Compose an email", "Compose a message",
)


def has_translation_artifact(text: str) -> bool:
    return any(p.search(text) for p in _TRANSLATION_PATTERNS)


def has_strong_math_signal(text: str) -> bool:
    if _MATH_LATEX_RE.search(text):
        return True
    if _MATH_INSTRUCTION_RE.search(text) and _MATH_OBJECT_RE.search(text):
        return True
    if _MATH_INSTRUCTION_RE.search(text) and re.search(r"\d", text):
        return True
    return False


def has_deep_math_signal(text: str) -> bool:
    """Stricter math check for STEM-keyword contexts.

    Physics/chemistry problems routinely use LaTeX for variable names and
    words like 'determine'/'calculate' with numeric values.  Only count a
    genuine math signal when the text mentions a *math object* (equation,
    polynomial, matrix, prime, etc.) alongside a math instruction.
    """
    return bool(_MATH_INSTRUCTION_RE.search(text) and _MATH_OBJECT_RE.search(text))


_ROLE_PATTERNS = [
    re.compile(r"\byou are (?:a |an |the )?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|who\b|$)", re.I),
    re.compile(r"\byou'?re (?:a |an |the )?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|who\b|$)", re.I),
    re.compile(r"\b(?:your|my) role is (?:a |an |the |to be (?:a |an )?)?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|$)", re.I),
    re.compile(r"\bact(?:ing)? as (?:a |an |the )?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|$)", re.I),
    re.compile(r"\bplay(?:ing)? the role of (?:a |an |the )?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|$)", re.I),
    re.compile(r"\b(?:imagine|pretend|assume|suppose) (?:you(?:'re| are)|that you(?:'re| are)|being) (?:a |an |the )?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|$)", re.I),
    re.compile(r"\brespond (?:as|like) (?:a |an )?(\w[\w\s]{0,30}?)(?:\.|,|\band\b|$)", re.I),
    re.compile(r"\bas (?:a |an )(\w[\w\s]{0,30}?),?\s*(?:your|you should|please|answer|help|solve|write|create|explain)", re.I),
]
_CHAT_TRANSCRIPT_RE = re.compile(
    r"^\s*(?:system|user|assistant)\s*:|<\s*(?:system|user|assistant|think)\s*>|"
    r"\b(?:follow|obey)\s+(?:the )?system rules\b",
    re.IGNORECASE | re.MULTILINE,
)
_MATH_ROLE_WHITELIST = re.compile(
    r"\b(?:math(?:ematic(?:s|ian))?|statistic(?:s|ian)|actuary|tutor"
    r"|professor|teacher|instructor|lecturer|solver|logician)\b",
    re.IGNORECASE,
)


def is_non_math_roleplay(text: str) -> bool:
    for pattern in _ROLE_PATTERNS:
        m = pattern.search(text)
        if m:
            role = m.group(1)
            if not _MATH_ROLE_WHITELIST.search(role):
                return True
    return False


def has_non_math_signal(text: str) -> bool:
    return bool(
        _NON_MATH_CODE_RE.search(text)
        or _NON_MATH_GENERAL_RE.search(text)
        or is_non_math_roleplay(text)
        or _CHAT_TRANSCRIPT_RE.search(text)
    )


_STRONG_CODE_RE = re.compile(
    r"\bwrite (?:a |an )?(?:python|java|c\+\+|rust|go|ruby|php) (?:program|function|class|script|method)\b|"
    r"\b(?:python|java|c\+\+|rust|go|ruby|php) (?:program|function|class|script) (?:that|which|to)\b|"
    r"\bcreate (?:a |an )?(?:python|java|c\+\+|rust|go) (?:program|function|class|script|method)\b|"
    r"\bimplement (?:a |an )?(?:python|java|c\+\+|rust|go) (?:program|function|class|script|method)\b|"
    r"\bsolve the following (?:coding|programming) problem\b|"
    r"```(?:python|java|javascript|typescript|ruby|php|cpp|c\+\+|rust|go|"
    r"swift|kotlin|scala|sql|bash|shell|csharp|erlang|haskell|lua|perl|"
    r"vb|visual basic|powershell)\b|"
    r"\bport this code\b|\btranslate this code\b|\bconvert this code\b|"
    r"\bpredict the output of\b.*\bcode\b|"
    r"\bgiven (?:the |this )?code\b.*\b(?:output|what (?:does|will|is))\b|"
    r"\bfunction '?\w+'?\s*,?\s*which manipulates\b|"
    r"\bcompetitive programming problem\b|"
    r"\bmodify (?:a |the )?code file\b|"
    r"\breturn the adapted code\b",
    re.IGNORECASE,
)


def has_strong_code_signal(text: str) -> bool:
    """Unambiguous coding task — overrides math signal."""
    return bool(_STRONG_CODE_RE.search(text))


def has_non_math_stem_signal(text: str) -> bool:
    return bool(_NON_MATH_STEM_RE.search(text))


def classify_non_math_signal(text: str) -> str:
    code = bool(_NON_MATH_CODE_RE.search(text))
    general = bool(_NON_MATH_GENERAL_RE.search(text))
    if code and general:
        return "both"
    if code:
        return "code"
    if general:
        return "general"
    return ""


def has_programming_prefix(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in PROGRAMMING_PREFIXES)


def is_dolphin_non_math(full_text: str) -> bool:
    return not has_strong_math_signal(full_text) or has_non_math_signal(full_text)


@register_processor("math_filter")
class MathFilterProcessor(RecordProcessor):
    name = "math_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
        cot_text = "\n".join(
            str(item.get("reasoning", ""))
            for item in dict(record.distillation).values()
            if isinstance(item, dict)
        ).strip()
        text = f"{record.question}\n{cot_text}".strip()
        has_translation = has_translation_artifact(text)
        has_math = has_strong_math_signal(text)
        has_deep_math = has_deep_math_signal(text)
        has_nonmath = has_non_math_signal(text)
        has_stem = has_non_math_stem_signal(text)
        has_code = has_strong_code_signal(text)
        should_drop = (
            has_translation
            or has_code
            or (has_nonmath and not has_deep_math)
            or (has_stem and not has_deep_math)
            or (has_stem and has_nonmath)
            or has_programming_prefix(text)
            or is_non_math_roleplay(text)
        )
        if should_drop:
            updated = record.clone(training_phase="drop", filter_tag="non_math_signal")
            updated.add_trace(stage="clean", processor=self.name, status="dropped", reason_code="non_math_signal")
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code="non_math_signal",
            )
        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
