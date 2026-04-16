import 'package:google_generative_ai/google_generative_ai.dart';

/// A single chat message — either from the user or from Gemini.
class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

/// Wraps a Gemini chat session. Re-create when the API key changes.
class ChatService {
  final List<ChatMessage> messages = [];

  late final ChatSession _chat;

  ChatService(String apiKey) {
    final model = GenerativeModel(
      model: 'gemini-1.5-flash',
      apiKey: apiKey,
      systemInstruction: Content.system(
        'You are a physiotherapy assistant. '
        'The user may share range of motion (ROM) measurements, describe movement '
        'limitations, or ask general physiotherapy questions. '
        'Provide concise, clinically relevant assessments. '
        'Use bullet points for structured information. '
        'Always remind the user to consult a licensed physiotherapist for diagnosis.',
      ),
    );
    _chat = model.startChat();
  }

  /// Send a message and return Gemini\'s reply text.
  Future<String> send(String userText) async {
    messages.add(ChatMessage(text: userText, isUser: true));
    final response = await _chat.sendMessage(Content.text(userText));
    final reply = response.text ?? '(no response)';
    messages.add(ChatMessage(text: reply, isUser: false));
    return reply;
  }

  void clear() => messages.clear();
}
