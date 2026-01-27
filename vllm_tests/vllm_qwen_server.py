from vllm import LLM, EngineArgs

from vllm.utils.argparse_utils import FlexibleArgumentParser

MAX_TOKENS = 32768 

def create_parser(): 
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    return parser

def create_qwen_server_llm(temperature: float = 0.0) -> LLM:
    model_name = 'Qwen/Qwen-0.6B'
    engine_args = EngineArgs(
        model=model_name,
        temperature=temperature,
        top_p=1.0
    )
    llm = LLM(engine_args)
    return llm

def main():
    prompt = "Tell me a joke about cats."
    parser = create_parser()
    args = parser.parse_args()
    llm = create_qwen_server_llm(args.temperature)
    answer = llm.generate(prompts=[prompt], max_tokens=MAX_TOKENS)
    print('answer:', answer)

if __name__ == "__main__":
    main()