// import { useEffect, useState } from "react";
// import { connect } from "react-redux";
// import { useHistory } from "react-router-dom";
// import '../styles/createProjectBtn.css';


// const CreateProjectButton = (props) => {

//     let [message, set_message] = useState("");
//     let { auth, dispatch } = props;
//     let history = useHistory();
//     let [newProjectDetails, setNewProjectDetails] = useState({
//         title:"",
//         description:""
//     });

//     useEffect(() => { 
//         setNewProjectDetails({
//             ...newProjectDetails,
//             title:"",
//             description:""
//         });
//     }, [auth]);

//     let handleInputState = () => {
//         setNewProjectDetails({
//             ...newProjectDetails,
//             title:"",
//             description:""
//         });
//     }

//     let handleCreationOfProject = async (e) => {
//         e.preventDefault();
//         console.log("new project created successfully");
//         let res = await doCreateProject(
//             {
//                 title:newProjectDetails.title,
//                 description:newProjectDetails.description
//             },
//             auth.token
//         );

//         set_message(res.message);
//     };

//     let handleNewProjectTitle = (e) =>{
//         setNewProjectDetails({ ...newProjectDetails, title:e.target.value});
//     }

//     let handleNewProjectDescription = (e) =>{
//         setNewProjectDetails({ ...newProjectDetails, description:e.target.value});
//     }

    
//     let createProjectButton = (
//         <div>
//             <button type="button" className="btn btn-primary" data-bs-toggle="modal" data-bs-target="#exampleModal" data-bs-whatever="@mdo" onClick={handleInputState}>New Project</button>

//             <div className="modal fade" id="exampleModal" tabIndex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
//                 <div className="modal-dialog">
//                     <div className="modal-content">
//                     <div className="modal-header">
//                         <h5 className="modal-title" id="exampleModalLabel">Enter Details to create new project</h5>
//                         <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
//                     </div>
//                     <div className="modal-body">
//                         <form onSubmit={(e) => handleCreationOfProject(e)}>
//                             <div className="mb-3">
//                                 <label htmlFor="project-title" className="col-form-label">Title Of Project:</label>
//                                 <input
//                                     type="text"
//                                     className="form-control"
//                                     id="project-title"
//                                     onChange={(e) => handleNewProjectTitle(e)}
//                                     value={newProjectDetails.title}
//                                 ></input>
//                             </div>
//                             <div className="mb-3">
//                                 <label htmlFor="project-description" className="col-form-label">Description Of Project:</label>
//                                 <textarea
//                                     className="form-control"
//                                     id="project-description"
//                                     onChange={(e) => handleNewProjectDescription(e)}
//                                     value={newProjectDetails.description}
//                                 ></textarea>
//                             </div>

//                             <div className="modal-footer">
//                                 <button type="button" className="btn btn-secondary" data-bs-dismiss="modal" >Close</button>
//                                 <button type="submit" className="btn btn-primary" data-bs-dismiss="modal" >Create Project</button>
//                             </div>

//                         </form>
//                     </div>
                    
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );

//     return createProjectButton;

// }

// let mapStateToProps = (state) => ({
//     auth: state.auth,
// });


// export default connect(mapStateToProps)(CreateProjectButton);