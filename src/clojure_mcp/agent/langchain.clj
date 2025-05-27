(ns clojure-mcp.agent.langchain
  (:require
    [clojure-mcp.agent.langchain.schema :as schema]
    [clojure.data.json :as json]
    [clojure.string :as string]
    [clojure.tools.logging :as log])
  (:import
    [dev.langchain4j.agent.tool ToolExecutionRequest]
    [dev.langchain4j.agent.tool ToolSpecification #_ToolParameter]
    [dev.langchain4j.data.message SystemMessage UserMessage TextContent ]
    [dev.langchain4j.memory ChatMemory]
    [dev.langchain4j.memory.chat MessageWindowChatMemory]
   ;; LangChain4j Model classes (using Anthropic as an example)
    [dev.langchain4j.model.anthropic
     AnthropicChatModel
     AnthropicStreamingChatModel
     AnthropicChatModelName]
    [dev.langchain4j.model.chat.request ChatRequest ToolChoice]
    [dev.langchain4j.model.chat.request.json JsonObjectSchema]
    [dev.langchain4j.model.googleai
     GoogleAiGeminiChatModel]
    [dev.langchain4j.model.openai
     OpenAiChatModel
     OpenAiChatRequestParameters
     OpenAiChatModelName]

   ;; LangChain4j Core and Service classes
    [dev.langchain4j.service AiServices MemoryId]
    [dev.langchain4j.service TokenStream]
    [dev.langchain4j.service.tool ToolExecutor ToolExecution]

   ;; Java Time API
    [java.time LocalTime LocalDate ZoneId]

    [java.util.function Consumer Function]))

(def default-max-memory 100)

#_(defn create-model-claude-3-7 []
    (-> (AnthropicChatModel/builder)
        (.apiKey (System/getenv "ANTHROPIC_API_KEY"))
        (.modelName AnthropicChatModelName/CLAUDE_3_7_SONNET_20250219)
        #_(.modelName AnthropicChatModelName/CLAUDE_3_5_SONNET_20241022)
        (.logRequests true)
        (.logResponses true)))

#_(defn create-model-claude-3-5 []
    (-> (AnthropicChatModel/builder)
        (.apiKey (System/getenv "ANTHROPIC_API_KEY"))
        (.modelName AnthropicChatModelName/CLAUDE_3_5_SONNET_20241022)
        #_(.modelName AnthropicChatModelName/CLAUDE_3_5_SONNET_20241022)
        (.logRequests true)
        (.logResponses true)))

#_(defn create-model-gemini []
    (-> (GoogleAiGeminiChatModel/builder)
        (.apiKey (System/getenv "GEMINI_API_KEY"))
        (.modelName "gemini-2.5-pro-preview-03-25")
        #_(.logRequests true)
        #_(.logResponses true)))

#_(defn create-model-gemini-2-0-flash []
    (-> (GoogleAiGeminiChatModel/builder)
        (.apiKey (System/getenv "GEMINI_API_KEY"))
        (.modelName "gemini-2.0-flash")
        #_(.logRequests true)
        #_(.logResponses true)))

;; simple API as we don't really need more right now

(defn create-gemini-model [model-name]
  (-> (OpenAiChatModel/builder)
      (.baseUrl "https://generativelanguage.googleapis.com/v1beta/openai/")
      (.apiKey (System/getenv "GEMINI_API_KEY"))
      (.modelName model-name)))

(defn create-openai-model [model-name]
  (-> (OpenAiChatModel/builder)
      (.apiKey (System/getenv "OPENAI_API_KEY"))
      (.baseUrl "http://192.168.50.2:5000/v1")
      (.modelName "qwen3-32b")))

;; reasoning not supported yet??
;; Langchain Anthropic client is unstable, using OPENAI api is better but
;; can't seem to find how to add request parameters for thinking to it
(defn create-anthropic-model [model-name]
  (-> (OpenAiChatModel/builder)
      (.baseUrl "https://api.anthropic.com/v1/")
      (.apiKey (System/getenv "ANTHROPIC_API_KEY"))
      (.modelName model-name)))

(defn default-request-parameters [model-builder configure-fn]
  (.defaultRequestParameters model-builder
    (.build (configure-fn (OpenAiChatRequestParameters/builder)))))

(defn reasoning-effort [request-params-builder reasoning-effort]
  (assert (#{:low :medium :high} reasoning-effort))
  (.reasoningEffort request-params-builder (name reasoning-effort)))

(defn max-output-tokens [request-params-builder max-output-tokens]
  (.maxOutputTokens request-params-builder (int max-output-tokens)))

(defn reasoning-agent-model []
  (cond
    (System/getenv "GEMINI_API_KEY")
    (create-gemini-model "gemini-2.5-flash-preview-05-20")
    (System/getenv "OPENAI_API_KEY")
    (create-openai-model "o4-mini")
    :else nil))

(defn agent-model []
  (cond
    (System/getenv "GEMINI_API_KEY")
    (create-gemini-model "gemini-2.5-flash-preview-05-20")
    (System/getenv "OPENAI_API_KEY")
    (create-openai-model "o4-mini")
    (System/getenv "ANTHROPIC_API_KEY")
    (create-anthropic-model AnthropicChatModelName/CLAUDE_3_7_SONNET_20250219)
    :else nil))

#_(-> (agent-model)
      (default-request-parameters #(max-output-tokens % 4096))
      (.build))

(defn chat-memory
  ([]
   (chat-memory default-max-memory))
  ([size]
   (MessageWindowChatMemory/withMaxMessages size)))

;; ------------------------------------------------------------
;; converting tools
;; ------------------------------------------------------------

(defn is-well-formed-json? [json-string]
  (try
    {:result (json/read-str json-string :key-fn keyword)}
    (catch Exception _ false)))

(defn registration-map->tool-executor [{:keys [tool-fn]}]
  (reify ToolExecutor
    (execute [_this request memory-id]
      (let [tool-name (.name request)
            arg-str (.arguments request)]

        (if-let [arg-result (is-well-formed-json? arg-str)]
          (try
            (log/info (str "Calling tool" (pr-str {:tool-name tool-name
                                                   :arg-result (:result arg-result)})))
            (let [callback-result (promise)]
              (tool-fn nil
                (:result arg-result)
                (fn [result error]
                  (deliver callback-result
                    (if error
                      (str "Tool Error: " (string/join "\n" result))
                      (if (sequential? result)
                        (string/join "\n\n" result)
                        (str result))))))
              @callback-result)
            (catch Exception e
              (str "Error executing tool '" tool-name "': " (.getMessage e)
                   (with-out-str (clojure.pprint/pprint (Throwable->map e))))))
          (str "ERROR: Arguments provided to the tool call were malformed.\n=====\n" arg-str "\n=====\n"))))))

(defn registration-map->tool-specification [{:keys [name description schema]}]
  {:pre [(or (string? schema) (map? schema))
         (string? name) (string? description)]}
  (-> (ToolSpecification/builder)
      (.name name)
      (.description description)
      (.parameters (schema/edn->sch
                     (if (string? schema)
                       (json/read-str schema :key-fn keyword)
                       schema)))
      (.build)))

(defn convert-tools [registration-maps]
  (into {}
    (map (juxt registration-map->tool-specification
           registration-map->tool-executor)
      registration-maps)))

(defn chat-request [message & {:keys [system-message tools require-tool-choice]}]
  ;; ChatResponse response = model.chat(request);
  ;;AiMessage aiMessage = response.aiMessage();
  (cond-> (ChatRequest/builder)
          system-message (.messages (list (SystemMessage. system-message)
                                      (UserMessage. message)))
          (not system-message) (.messages (list (UserMessage. message)))
          (not-empty tools) (.toolSpecifications (map registration-map->tool-specification tools))
    ;; TODO on next langchain bump
    ;; require-tool-choice (.toolChoice ToolChoice/REQUIRED)
          :else (.build)))

(definterface AiService
  (^String chat [^String userMessage])
  (^String chat [^dev.langchain4j.data.message.UserMessage userMessage]))

(defn create-service [klass {:keys [model memory tools system-message]}]
  (-> (AiServices/builder klass) ; Use the interface defined above
      (.chatModel model)
      (.systemMessageProvider
        (reify Function
          (apply [this mem-id]
            system-message)))
      (cond->
        tools (.tools (convert-tools tools)))
      (.chatMemory memory)))

(comment
  (def test-tool
    {:name "hello"
     :description "says hello"
     :schema
     (json/json-str {:type :object
                     :properties
                     {:nm {:type :string
                           :description "Name to say hi to"}}
                     :required [:nm]})
     :tool-fn (fn [_ {:keys [nm]} callback]
                (callback [(str "Hello " nm "!")] false))})
  #_(create-service AiService {})

  (let [exec (registration-map->tool-executor test-tool)]
    (.execute exec (-> (ToolExecutionRequest/builder)
                       (.name "hello")
                       (.arguments (json/json-str {:nm "Joey"}))
                       (.build))
      "asdf"))

  )

;; keep this example
#_(definterface StreamingAiService
    (^dev.langchain4j.service.TokenStream chat [^String userMessage])
    (^dev.langchain4j.service.TokenStream chat [^dev.langchain4j.data.message.UserMessage userMessage]))
