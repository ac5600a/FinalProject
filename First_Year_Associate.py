import os
import glob
import subprocess
import requests
from pypdf import PdfReader

API_URL = "https://ollama.com/api/chat"
HEADERS = {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}

read_file_tool_schema = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Use this when the user wants read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "the path of the file to read"},
            },
            "required": ["path"]
        }
    }
}

list_files_schema = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "Use this when the user wants to list files in the current working directory by default, or in a named directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "the path of the directory on which to list the files."}
            },
            "required": []
        }
    }
}

write_files_schema = {
    "type": "function",
    "function": {
        "name": "write_files",
        "description": "Use this when the user wants to write content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "the path with filename and extension of the file to write."},
                "content": {"type": "string", "description": "the content to write to the file."}
            },
            "required": ["path", "content"]
        }
    }
}

run_python_schema = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Use this when the user wants to run Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "the path of the python code to run."}
            },
            "required": ["path"]
        }
    }
}

search_files_schema = {
    "type": "function",
    "function": {
        "name": "search_case_documents",
        "description": "Searches all case documents, includeing PDF and TXT files, for a word or phrase.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string"
                }
            },
            "required": ["search_term"]
        }
    }
}

def agent(messages):
    tools = [
        read_file_tool_schema, list_files_schema, write_files_schema, run_python_schema, search_files_schema
    ]

    payload = {
        "model": "gpt-oss:120b",
        "messages": messages,
        "stream": False,
        "tools": tools
    }
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    message = data["message"]
    return message

while True:
    user_question = input("\nWhat would you like to know? ")

    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages = [
        {"role": "system", 
        "content": """
            You are a junior associate at a law firm working on
            United States v. Apex Technologies, Inc.

            Your responsibilities include:
            1. Reviewing and summarizing case documents.
            2. Identifying relevant antitrust issues under Section 1
            and Section 2 of the Sherman Act.
            3. Finding relevant evidence in emails, contracts,
            depositions, and pleadings.
            4. Identifying inconsistencies between documents.
            5. Drafting preliminary legal research and case summaries.
            6. Preparing work product for review by a supervising partner.
            7. Answering questions about the case and the documents.
            8. Drafting and editing any legal documents, including pleadings, motions, and discovery requests.

            Use the available tools to read, search, and organize
            documents in the case file.

            Be accurate, professional, and concise.
                When answering questions:
                    1. Answer the user's specific question directly.
                    2. Use only the information necessary to answer.
                    3. Avoid lengthy introductions and unnecessary background.
                    4. Use bullet points when they improve readability.
                    5. Do not repeat information unless necessary.
                    6. Do not describe every tool call or document reviewed.
                    7. If the user asks a simple factual question, provide a short answer.
                    8. If the user requests legal analysis or drafting, provide
                    more detailed responses appropriate to the task.
            Distinguish established facts from allegations and
            potential legal conclusions.
            Never invent facts or evidence.
            When working with case documents:
                1. Always use search_case_documents to identify relevant documents.
                2. Only use exact file paths returned by search_case_documents
                or list_files.
                3. Never invent, modify, or guess a file path.
                4. Use read_file to retrieve the actual contents of relevant documents.
                5. When reviewing PDFs, use the read_file tool just as you
                would for TXT files.
                6. If a document cannot be located, explain that rather than
                inventing its contents.
                7. Review multiple relevant documents when necessary to
                answer the user's question.
            """
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    tool_called = True

    while tool_called:
        message = agent(messages)
        tool_called = "tool_calls" in message

        if tool_called:
            messages.append(message)
            tool_calls = message["tool_calls"]
            tool_call = tool_calls[0]

            function_name = tool_call["function"]["name"]
            function_arguments = tool_call["function"]["arguments"]

            if function_name == "list_files":
                path = "."
                if "path" in function_arguments:
                    path = function_arguments["path"]
                files = glob.glob(f"{path}/*")
                output = files

            if function_name == "read_file":
                path = function_arguments["path"]

                if not os.path.isfile(path):
                    output = f"Error: File not found at {path}. Use search_case_documents or list_files to find the correct file path."

                elif path.lower().endswith(".pdf"):
                    reader = PdfReader(path)
                    file_content = "\n".join(
                        page.extract_text() or ""
                        for page in reader.pages
                    )
                    output = file_content

                else:
                    with open(path, "r", encoding="utf-8") as file:
                        output = file.read()

            if function_name == "write_files":
                path = function_arguments["path"]
                content = function_arguments["content"]
                
                with open(path, "w") as file:
                    file.write(content)
                
                output = f"File written to {path} with content: {content}"

            if function_name == "run_python":
                path = function_arguments["path"]
                result = subprocess.run(["python", path], capture_output=True, text=True, check=True)
                output = f" output of python {path}"

            if function_name == "search_case_documents":
                search_term = function_arguments["search_term"]
                output = ""

                files = glob.glob(
                    os.path.join("input", "**", "*.txt"),
                    recursive=True
                )
                files += glob.glob(
                    os.path.join("input", "**", "*.pdf"),
                    recursive=True
                )

                for path in files:
                    try:
                        if path.lower().endswith(".pdf"):
                            reader = PdfReader(path)
                            content = "\n".join(
                                page.extract_text() or ""
                                for page in reader.pages
                            )
                        else:
                            with open(path, "r", encoding="utf-8") as file:
                                content = file.read()

                        if search_term.lower() in content.lower():
                            output += f"\nMatching document: {path}\n"

                    except Exception as e:
                        print(f"Loading: Error reading {path}: {e}")

                if output == "":
                    output = f"No matches found for '{search_term}'."

                output = f"Search results for '{search_term}':\n{output}"

                print("Finished searching documents.")
             
            tool_message = {
                "role": "tool",
                "content": f"The output of tool {function_name} is: {output}"
            }
            messages.append(tool_message)
        else:
            ai_message = {
                "role":  message["role"],
                "content": message["content"]
            }
            messages.append(ai_message)

    last_message = messages[-1]

    print("\n" + last_message["content"])
