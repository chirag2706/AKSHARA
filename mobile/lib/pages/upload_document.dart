import 'package:akshara/providers/auth_provider.dart';
import 'package:akshara/utils/constants.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

class UploadDocument extends StatefulWidget {
  final String projectId;
  const UploadDocument({Key? key, required this.projectId}) : super(key: key);

  @override
  State<UploadDocument> createState() => _UploadDocumentState();
}

class _UploadDocumentState extends State<UploadDocument> {
  final ImagePicker _picker = ImagePicker();
  final _formKey = GlobalKey<FormState>();
  var title = '';
  var description = '';
  XFile? pickedImage;
  void selectImage(cond) async {
    if (cond) {
      pickedImage = await _picker.pickImage(source: ImageSource.gallery);
    } else {
      pickedImage = await _picker.pickImage(source: ImageSource.camera);
    }
  }

  void upload() async {
    var request = http.MultipartRequest(
        'POST', Uri.parse(AppConstants.baseUrl + "/documents"));
    request.fields['name'] = title;
    request.fields['description'] = description;
    request.fields['project'] = widget.projectId;
    request.files
        .add(await http.MultipartFile.fromPath('image', pickedImage!.path));
    request.headers['authtoken'] =
        Provider.of<AuthProvider>(context, listen: false).user.token.toString();
    var res = await request.send();
    if (res.statusCode == 200) {
      Navigator.of(context).pop();
    } else {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text("Request failed")));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: <Widget>[
            const Text(
              'Upload Document',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 40,
              ),
            ),
            const SizedBox(
              height: 60,
            ),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  TextFormField(
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter title';
                      }
                      return null;
                    },
                    maxLines: 1,
                    onChanged: (val) {
                      title = val;
                    },
                    decoration: InputDecoration(
                      hintText: 'Name',
                      prefixIcon: const Icon(Icons.person),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  TextFormField(
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter description';
                      }
                      return null;
                    },
                    minLines: 4,
                    maxLines: 6,
                    onChanged: (val) {
                      description = val;
                    },
                    decoration: InputDecoration(
                      hintText: 'Enter Description',
                      prefixIcon: const Icon(Icons.email),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                          onPressed: () {
                            selectImage(true);
                          },
                          icon: const Icon(Icons.photo_album)),
                      IconButton(
                          onPressed: () {
                            selectImage(false);
                          },
                          icon: const Icon(Icons.photo_camera)),
                    ],
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  ElevatedButton(
                    onPressed: () {
                      if (_formKey.currentState!.validate()) {
                        upload();
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.fromLTRB(40, 15, 40, 15),
                    ),
                    child: const Text(
                      'Upload document',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
